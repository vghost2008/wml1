#coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
import wnn
import functools
import wtfop.wtfop_ops as wop
import object_detection2.bboxes as odbox
from object_detection2.standard_names import *
import wmodule
from object_detection2.modeling.onestage_heads.onestage_tools import *
from object_detection2.datadef import *
from object_detection2.config.config import global_cfg
import object_detection2.wlayers as odl
from object_detection2.modeling.build import HEAD_OUTPUTS
from object_detection2.data.dataloader import DataLoader
import wsummary


'''
Use GIOU loss instated the official huber loss for box regression.
'''
@HEAD_OUTPUTS.register()
class RetinaNetGIOUOutputs(wmodule.WChildModule):
    def __init__(
        self,
        cfg,
        parent,
        box2box_transform,
        anchor_matcher,
        pred_logits,
        pred_anchor_deltas,
        anchors,
        gt_boxes=None,
        gt_labels=None,
        gt_length=None,
        max_detections_per_image=100,
        **kwargs,
    ):
        """
        Args:
            cfg: Only the child part
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for
                anchor-proposal transformations.
            anchor_matcher (Matcher): :class:`Matcher` instance for matching anchors to
                ground-truth boxes; used to determine training labels.
            images (ImageList): :class:`ImageList` instance representing N input images
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for anchors.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, A*4, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
            anchors (list[list[Boxes]]): A list of N elements. Each element is a list of L
                Boxes. The Boxes at (n, l) stores the entire anchor array for feature map l in image
                n (i.e. the cell anchors repeated over all locations in feature map (n, l)).
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image
                boundary by more than boundary_thresh are not used in training. Set to a very large
                number or < 0 to disable this behavior. Only needed in training.
            gt_boxes (list[Boxes], optional): A list of N elements. Element i a Boxes storing
                the ground-truth ("gt") boxes for image i.
        """
        super().__init__(cfg,parent=parent,**kwargs)
        self.num_classes      = cfg.NUM_CLASSES
        self.focal_loss_alpha         = cfg.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma         = cfg.FOCAL_LOSS_GAMMA
        self.topk_candidates          = cfg.TOPK_CANDIDATES_TEST
        self.score_threshold          = cfg.SCORE_THRESH_TEST
        self.nms_threshold            = cfg.NMS_THRESH_TEST
        self.max_detections_per_image = max_detections_per_image
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.pred_logits = pred_logits
        self.pred_anchor_deltas = pred_anchor_deltas
        self.anchors_lens = [tf.shape(x)[0] for x in anchors]


        anchors = tf.concat(anchors,axis=0)
        anchors = tf.expand_dims(anchors,axis=0)
        self.anchors = anchors
        self.gt_boxes = gt_boxes
        self.gt_labels = gt_labels
        self.gt_length = gt_length
        self.num_feature_maps = len(pred_logits)
        self.mid_results = {}

    def _get_ground_truth(self):
        """
        Returns:
            gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """
        res = self.anchor_matcher(self.anchors,self.gt_boxes,self.gt_labels,self.gt_length,
                                  boxes_len = self.anchors_lens)

        gt_objectness_logits_i, scores, indices  = res
        self.mid_results['anchor_matcher'] = res

        gt_anchor_deltas = wmlt.batch_gather(self.gt_boxes,indices)
        #gt_objectness_logits_i为相应anchor box的标签
        return gt_objectness_logits_i, gt_anchor_deltas

    @wmlt.add_name_scope
    def losses(self):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNetGIou.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`RetinaNetGIouHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        
        assert len(self.pred_logits[0].get_shape()) == 4,"error logits dim"
        assert len(self.pred_anchor_deltas[0].get_shape()) == 4,"error anchors dim"
        
        gt_classes,gt_anchors_deltas = self._get_ground_truth()
        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            self.pred_logits, self.pred_anchor_deltas, self.num_classes
        )  # Shapes: (N, R, K) and (N, R, 4), respectively.


        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes > 0)
        num_foreground = tf.reduce_sum(tf.cast(foreground_idxs,tf.int32))

        gt_classes_target = tf.boolean_mask(gt_classes,valid_idxs)
        gt_classes_target = tf.one_hot(gt_classes_target,depth=self.num_classes+1)
        gt_classes_target = gt_classes_target[:,1:]#RetinaNetGIou中没有背景, 因为背景index=0, 所以要在one hot 后去掉背景
        pred_class_logits = tf.boolean_mask(pred_class_logits,valid_idxs)

        # logits loss
        loss_cls = tf.reduce_sum(wnn.sigmoid_cross_entropy_with_logits_FL(
            labels = gt_classes_target,
            logits = pred_class_logits,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
        )) / tf.cast(tf.maximum(1, num_foreground),tf.float32)

        # regression loss
        pred_anchor_deltas = tf.boolean_mask(pred_anchor_deltas,foreground_idxs)
        gt_anchors_deltas = tf.boolean_mask(gt_anchors_deltas,foreground_idxs)
        B,X = wmlt.combined_static_and_dynamic_shape(foreground_idxs)
        anchors = tf.tile(self.anchors,[B,1,1])
        anchors = tf.boolean_mask(anchors,foreground_idxs)
        box = self.box2box_transform.apply_deltas(pred_anchor_deltas,anchors)
        reg_loss_sum = 1.0-odl.giou(box,gt_anchors_deltas)
        loss_box_reg = tf.reduce_sum(reg_loss_sum) / tf.cast(tf.maximum(1, num_foreground),tf.float32)

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    @wmlt.add_name_scope
    def inference(self, inputs,box_cls, box_delta,anchors):
        """
        Arguments:
            inputs: same as RetinaNet.forward's batched_inputs
            box_cls: list of Tensor, Tensor's shape is [B,H,W,A*num_classes]
            box_delta: list of Tensor, Tensor's shape is [B,H,W,A*4]
            anchors: list of Tensor, Tensor's shape is [X,4]( X=H*W*A)
        Returns:
            results:
            RD_BOXES: [B,N,4]
            RD_LABELS: [B,N]
            RD_PROBABILITY:[ B,N]
            RD_LENGTH:[B]
        """
        assert len(anchors[0].get_shape()) == 2, "error anchors dims"
        assert len(box_cls[0].get_shape()) == 4, "error box cls dims"
        assert len(box_delta[0].get_shape()) == 4, "error box delta dims"

        anchors_size = [tf.shape(x)[0] for x in anchors]
        anchors = tf.concat(anchors,axis=0)


        box_cls = [reshape_to_N_HWA_K(x,self.num_classes) for x in box_cls]
        box_delta = [reshape_to_N_HWA_K(x,4) for x in box_delta]
        box_cls = tf.concat(box_cls,axis=1)
        box_delta = tf.concat(box_delta,axis=1)

        results = wmlt.static_or_dynamic_map_fn(lambda x:self.inference_single_image(x[0],x[1],anchors,anchors_size),elems=[box_cls,box_delta],
                                                dtype=[tf.float32,tf.int32,tf.float32,tf.int32],
                                                back_prop=False)
        outdata = {RD_BOXES:results[0],RD_LABELS:results[1],RD_PROBABILITY:results[2],RD_LENGTH:results[3]}
        if global_cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
            wsummary.detection_image_summary(images=inputs[IMAGE],
                                             boxes=outdata[RD_BOXES], classes=outdata[RD_LABELS],
                                             lengths=outdata[RD_LENGTH],
                                             scores=outdata[RD_PROBABILITY],
                                              name="RetinaNetGIou_result",
                                             category_index=DataLoader.category_index)
        return outdata

    @wmlt.add_name_scope
    def inference_single_image(self, box_cls, box_delta, anchors,anchors_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls;[WxHxA(concat),K]
            box_delta [WxHxA(concat),box_dim]
            anchors [WxHxA(concat),box_dim]
            anchors_size: anchors'size per level
        Returns:
            Same as `inference`, but for only one image.
        """
        
        assert len(box_cls.get_shape())==2,"Error box cls tensor shape"
        assert len(box_delta.get_shape())==2,"Error box delta tensor shape"
        assert len(anchors.get_shape())==2,"Error anchors tensor shape"
        
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        box_cls = tf.split(box_cls,num_or_size_splits=anchors_size)
        box_delta = tf.split(box_delta,num_or_size_splits=anchors_size)
        anchors = tf.split(anchors,num_or_size_splits=anchors_size)
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            box_cls_i = tf.nn.sigmoid(tf.reshape(box_cls_i,[-1]))

            # Keep top k top scoring indices only.
            num_topk = tf.minimum(self.topk_candidates, tf.shape(box_reg_i)[0])
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = tf.nn.top_k(box_cls_i,num_topk)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = tf.greater(predicted_prob,self.score_threshold)
            predicted_prob = tf.boolean_mask(predicted_prob,keep_idxs)
            topk_idxs = tf.boolean_mask(topk_idxs,keep_idxs)

            #retinanet is cls agnostic
            #in this process type, a some anchor may belong to multi class
            #after nms this will be fixed
            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = tf.gather(box_reg_i,anchor_idxs)
            anchors_i = tf.gather(anchors_i,anchor_idxs)
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i)


            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all= [
            tf.concat(x,axis=0) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        x,y = wmlt.sort_data(key=scores_all,datas=[boxes_all,class_idxs_all])
        boxes_all,class_idxs_all = y
        scores_all,_ = x
        nms = functools.partial(wop.boxes_nms, threshold=self.nms_threshold, classes_wise=True,k=self.max_detections_per_image)
        boxes,labels,nms_idxs = nms(bboxes=boxes_all,classes=class_idxs_all)
        scores = tf.gather(scores_all,nms_idxs)

        candiate_nr = self.max_detections_per_image
        #labels = tf.Print(labels,[tf.shape(labels)],name="XXXXXXXXXXXXXXXXXXXX",summarize=100)
        labels = labels+1 #加上背景
        lens = tf.shape(labels)[0]
        boxes = tf.pad(boxes, paddings=[[0, candiate_nr - lens], [0, 0]])
        labels = tf.pad(labels, paddings=[[0, candiate_nr - lens]])
        scores = tf.pad(scores, paddings=[[0, candiate_nr - lens]])

        return [boxes,labels,scores,lens]

