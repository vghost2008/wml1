#coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
import wnn
import functools
import tfop
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
from itertools import count
import numpy as np


'''
Use GIOU loss instated the official huber loss for box regression.
'''
@HEAD_OUTPUTS.register()
class FCOSGIOUOutputsV2(wmodule.WChildModule):
    def __init__(
        self,
        cfg,
        parent,
        box2box_transform,
        pred_logits,
        pred_regression,
        pred_center_ness,
        gt_boxes=None,
        gt_labels=None,
        gt_length=None,
        max_detections_per_image=100,
        batched_inputs=None,
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
        assert len(cfg.SIZE_THRESHOLD) == len(pred_logits)-1,"Error size threshold num."
        self.num_classes      = cfg.NUM_CLASSES
        self.focal_loss_alpha         = cfg.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma         = cfg.FOCAL_LOSS_GAMMA
        self.topk_candidates          = cfg.TOPK_CANDIDATES_TEST
        self.score_threshold          = cfg.SCORE_THRESH_TEST
        self.nms_threshold            = cfg.NMS_THRESH_TEST
        self.max_detections_per_image = max_detections_per_image
        self.size_threshold = [0]+cfg.SIZE_THRESHOLD+[-1]
        self.box2box_transform = box2box_transform
        self.pred_logits = pred_logits
        self.pred_regression = pred_regression
        self.pred_center_ness = pred_center_ness

        self.gt_boxes = gt_boxes
        self.gt_labels = gt_labels
        self.gt_length = gt_length
        self.num_feature_maps = len(pred_logits)
        self.mid_results = {}
        self.batched_inputs = batched_inputs

    def _get_ground_truth(self):
        """
        Returns:
            gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """

        img_size = tf.shape(self.batched_inputs[IMAGE])[1:3]

        res_list = []

        for i,logits,regression,center_ness in zip(count(),self.pred_logits,self.pred_regression,self.pred_center_ness):
            res = self.box2box_transform.get_deltas(gboxes=self.gt_boxes,
                                                    glabels=self.gt_labels,
                                                    glength=self.gt_length,
                                                    min_size=self.size_threshold[i],
                                                    max_size=self.size_threshold[i+1],
                                                    fm_shape=tf.shape(logits)[1:3],
                                                    img_size=img_size)
            if global_cfg.GLOBAL.SUMMARY_LEVEL <= SummaryLevel.DEBUG:
                for k,v in res.items():
                    if len(v.get_shape()) == 3:
                        v = tf.expand_dims(v,axis=-1)
                    wsummary.feature_map_summary(v,k)
            res_list.append(res)

        return res_list

    @wmlt.add_name_scope
    def losses(self):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`FCOSGIou.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`FCOSGIouHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        
        assert len(self.pred_logits[0].get_shape()) == 4,"error logits dim"

        gt_results = self._get_ground_truth()
        loss_cls_list = []
        loss_regression_list = []
        total_num_foreground = []
        
        img_size = tf.shape(self.batched_inputs[IMAGE])[1:3]

        for i,gt_results_item in enumerate(gt_results):
            gt_classes = gt_results_item['g_classes']
            gt_boxes = gt_results_item['g_boxes']
            pred_class_logits = self.pred_logits[i]
            pred_regression = self.pred_regression[i]
            pred_center_ness = self.pred_center_ness[i]

            foreground_idxs = (gt_classes > 0)
            num_foreground = tf.reduce_sum(tf.cast(foreground_idxs,tf.int32))
            total_num_foreground.append(num_foreground)

            gt_classes_target = tf.one_hot(gt_classes,depth=self.num_classes+1)
            gt_classes_target = gt_classes_target[...,1:]

            #
            pred_center_ness = tf.expand_dims(tf.nn.sigmoid(pred_center_ness),axis=-1)
            wsummary.histogram_or_scalar(pred_center_ness,"center_ness")
            # logits loss
            loss_cls = tf.reduce_sum(wnn.sigmoid_cross_entropy_with_logits_FL(
                labels = gt_classes_target,
                logits = pred_class_logits,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
            )*pred_center_ness+tf.pow(1-pred_center_ness,2)/3)

            # regression loss
            pred_boxes = self.box2box_transform.apply_deltas(regression=pred_regression,img_size=img_size)
            if global_cfg.GLOBAL.SUMMARY_LEVEL <= SummaryLevel.DEBUG and gt_classes.get_shape().as_list()[0]>1:
                log_boxes = self.box2box_transform.apply_deltas(regression=gt_results_item['g_regression'],
                                                                img_size=img_size)
                log_boxes = odbox.tfabsolutely_boxes_to_relative_boxes(log_boxes,width=img_size[1],height=img_size[0])
                boxes1 = tf.reshape(log_boxes[1:2],[1,-1,4])
                wsummary.detection_image_summary(images=self.batched_inputs[IMAGE][1:2],
                                                 boxes=boxes1,
                                                 name="FCOSGIou_decode_test")
            pred_center_ness = tf.boolean_mask(pred_center_ness,foreground_idxs)
            pred_boxes = tf.boolean_mask(pred_boxes,foreground_idxs)
            gt_boxes = tf.boolean_mask(gt_boxes,foreground_idxs)
            wsummary.histogram_or_scalar(pred_center_ness,"center_ness_pos")
            reg_loss_sum = (1.0-odl.giou(pred_boxes,gt_boxes))
            wmlt.variable_summaries_v2(reg_loss_sum,f"giou_loss{i}")
            pred_center_ness = tf.squeeze(pred_center_ness,axis=-1)
            reg_loss_sum = reg_loss_sum*(pred_center_ness+1e-2)
            wmlt.variable_summaries_v2(reg_loss_sum,f"loss_sum{i}")
            loss_box_reg = tf.reduce_sum(reg_loss_sum)
            wmlt.variable_summaries_v2(loss_box_reg,f"box_reg_loss_{i}")

            loss_cls_list.append(loss_cls)
            loss_regression_list.append(loss_box_reg)

        total_num_foreground = tf.to_float(tf.maximum(tf.add_n(total_num_foreground),1))
        return {"fcos_loss_cls": tf.add_n(loss_cls_list)/total_num_foreground,
                "fcos_loss_box_reg": tf.add_n(loss_regression_list)/total_num_foreground}

    @wmlt.add_name_scope
    def inference(self, inputs,box_cls, box_regression,center_ness,nms=None,pad=True):
        """
        Arguments:
            inputs: same as FCOS.forward's batched_inputs
            box_cls: list of Tensor, Tensor's shape is [B,H,W,A*num_classes]
            box_delta: list of Tensor, Tensor's shape is [B,H,W,A*4]
        Returns:
            results:
            RD_BOXES: [B,N,4]
            RD_LABELS: [B,N]
            RD_PROBABILITY:[ B,N]
            RD_LENGTH:[B]
        """
        assert len(box_cls[0].get_shape()) == 4, "error box cls dims"
        assert len(box_regression[0].get_shape()) == 4, "error box delta dims"

        B,_,_,_ = wmlt.combined_static_and_dynamic_shape(box_regression[0])
        fm_sizes = [tf.shape(x)[1:3] for x in box_regression]
        box_cls = [reshape_to_N_HWA_K(x,self.num_classes) for x in box_cls]
        box_regression = [reshape_to_N_HWA_K(x,4) for x in box_regression]
        center_ness = [tf.reshape(x,[B,-1]) for x in center_ness]
        box_cls = tf.concat(box_cls,axis=1)
        box_regression = tf.concat(box_regression,axis=1)
        center_ness = tf.concat(center_ness,axis=1)

        results = wmlt.static_or_dynamic_map_fn(lambda x:self.inference_single_image(x[0],x[1],x[2],fm_sizes,nms=nms,pad=pad),
                                                elems=[box_cls,box_regression,center_ness],
                                                dtype=[tf.float32,tf.int32,tf.float32,tf.int32],
                                                back_prop=False)
        outdata = {RD_BOXES:results[0],RD_LABELS:results[1],RD_PROBABILITY:results[2],RD_LENGTH:results[3]}
        if global_cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
            wsummary.detection_image_summary(images=inputs[IMAGE],
                                             boxes=outdata[RD_BOXES], classes=outdata[RD_LABELS],
                                             lengths=outdata[RD_LENGTH],
                                             scores=outdata[RD_PROBABILITY],
                                              name="FCOSGIou_result",
                                             category_index=DataLoader.category_index)
        return outdata

    @wmlt.add_name_scope
    def inference_single_image(self, box_cls, box_regression, center_ness,fm_sizes,nms = None,pad=True):
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
        assert len(box_regression.get_shape())==2,"Error box delta tensor shape"

        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        sizes = [tf.reduce_prod(x) for x in fm_sizes]
        box_cls = tf.split(box_cls,num_or_size_splits=sizes)
        box_regression = tf.split(box_regression,num_or_size_splits=sizes)
        center_ness = tf.split(center_ness,num_or_size_splits=sizes)
        img_size = tf.shape(self.batched_inputs[IMAGE])[1:3]
        for fm_size,box_cls_i, box_reg_i, centern_ness_i in zip(fm_sizes,box_cls, box_regression, center_ness):

            boxes_i = self.box2box_transform.apply_deltas(regression=box_reg_i,img_size=img_size,
                                                             fm_size=fm_size)
            boxes_i = odbox.tfabsolutely_boxes_to_relative_boxes(boxes_i,
                                                                 width=img_size[1], height=img_size[0])

            # (HxWxAxK,)
            box_cls_i = tf.nn.sigmoid(tf.reshape(box_cls_i,[-1]))

            # Keep top k top scoring indices only.
            num_topk = tf.minimum(self.topk_candidates, tf.shape(box_reg_i)[0])
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = tf.nn.top_k(box_cls_i,num_topk)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            if self.score_threshold>1e-5:
                keep_idxs = tf.greater(predicted_prob,self.score_threshold)
                predicted_prob = tf.boolean_mask(predicted_prob,keep_idxs)
                topk_idxs = tf.boolean_mask(topk_idxs,keep_idxs)

            boxes_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            # filter out the proposals with low confidence score

            boxes_i = tf.gather(boxes_i,boxes_idxs)
            center_ness = tf.nn.sigmoid(tf.gather(centern_ness_i,boxes_idxs))
            # predict boxes


            boxes_all.append(boxes_i)
            scores_all.append(predicted_prob*center_ness)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all= [
            tf.concat(x,axis=0) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        x,y = wmlt.sort_data(key=scores_all,datas=[boxes_all,class_idxs_all])
        boxes_all,class_idxs_all = y
        scores_all,_ = x
        if nms is None:
            nms = functools.partial(tfop.boxes_nms, threshold=self.nms_threshold, classes_wise=True,k=self.max_detections_per_image)
        boxes,labels,nms_idxs = nms(bboxes=boxes_all,classes=class_idxs_all)
        scores = tf.gather(scores_all,nms_idxs)

        candiate_nr = self.max_detections_per_image
        #labels = tf.Print(labels,[tf.shape(labels)],name="XXXXXXXXXXXXXXXXXXXX",summarize=100)
        labels = labels+1 #加上背景
        lens = tf.shape(labels)[0]
        if pad:
            boxes = tf.pad(boxes, paddings=[[0, candiate_nr - lens], [0, 0]])
            labels = tf.pad(labels, paddings=[[0, candiate_nr - lens]])
            scores = tf.pad(scores, paddings=[[0, candiate_nr - lens]])

        return [boxes,labels,scores,lens]

