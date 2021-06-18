#coding=utf-8
import tensorflow as tf
from object_detection2.modeling.build import HEAD_OUTPUTS
import wsummary
import wml_tfutils as wmlt
import wnn
import functools
import tfop
import object_detection2.bboxes as odbox
from object_detection2.standard_names import *
import wmodule
from object_detection2.modeling.sampling import subsample_labels_by_negative_loss
from .onestage_tools import *
from object_detection2.datadef import *
from object_detection2.config.config import global_cfg
from object_detection2.data.dataloader import DataLoader

def ssd_losses(
    gt_logits,
    gt_anchor_deltas,
    pred_logits,
    pred_anchor_deltas,
):
    localization_loss = tf.losses.huber_loss(
        predictions=pred_anchor_deltas,
        labels=gt_anchor_deltas,
        reduction=tf.losses.Reduction.SUM,
        loss_collection=None
    )

    objectness_loss = tf.losses.sparse_softmax_cross_entropy(
        logits=pred_logits,
        labels=gt_logits,
        reduction=tf.losses.Reduction.SUM,
        loss_collection=None
    )
    return objectness_loss, localization_loss

@HEAD_OUTPUTS.register()
class SSDOutputs(wmodule.WChildModule):
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
        **kwargs,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for
                anchor-proposal transformations.
            anchor_matcher (Matcher): :class:`Matcher` instance for matching anchors to
                ground-truth boxes; used to determine training labels.
            images (ImageList): :class:`ImageList` instance representing N input images
            pred_logits (list[Tensor]): A list of L elements.
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
        self.num_classes      = cfg.MODEL.SSD.NUM_CLASSES
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.pred_logits = pred_logits
        self.pred_anchor_deltas = pred_anchor_deltas
        self.batch_size_per_image    = cfg.MODEL.SSD.BATCH_SIZE_PER_IMAGE
        self.positive_fraction       = cfg.MODEL.SSD.POSITIVE_FRACTION
        self.topk_candidates = cfg.MODEL.SSD.TOPK_CANDIDATES_TEST
        self.score_threshold = cfg.MODEL.SSD.SCORE_THRESH_TEST
        self.nms_threshold = cfg.MODEL.SSD.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

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
            gt_logits: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """
        res = self.anchor_matcher(self.anchors,self.gt_boxes,self.gt_labels,self.gt_length)
        gt_logits_i, scores, indices  = res
        self.mid_results['anchor_matcher'] = res

        gt_anchor_deltas = self.box2box_transform.get_deltas(self.anchors,self.gt_boxes,gt_logits_i,indices)
        #gt_logits_i为相应anchor box的标签
        return gt_logits_i, gt_anchor_deltas

    @wmlt.add_name_scope
    def losses(self):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`RetinaNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_logits = [reshape_to_N_HWA_K(x,K=self.num_classes+1) for x in self.pred_logits]
        pred_logits = tf.concat(pred_logits,axis=1)
        gt_logits,gt_anchors_deltas = self._get_ground_truth()
        pos_idx, neg_idx = subsample_labels_by_negative_loss(gt_logits,
                                            self.batch_size_per_image,
                                            tf.nn.softmax(pred_logits),
                                            self.num_classes,
                                            self.positive_fraction)

        batch_size = self.pred_logits[0].get_shape().as_list()[0]
        num_cell_anchors = self.pred_logits[0].get_shape().as_list()[-1]//(self.num_classes+1)
        box_dim = self.pred_anchor_deltas[0].get_shape().as_list()[-1]//num_cell_anchors
        pred_anchor_deltas = [tf.reshape(x,[batch_size,-1,box_dim]) for x in self.pred_anchor_deltas]
        pred_anchor_deltas = tf.concat(pred_anchor_deltas,axis=1)
        pred_logits = tf.reshape(pred_logits,[-1,self.num_classes+1])
        pred_anchor_deltas = tf.reshape(pred_anchor_deltas,[-1,box_dim])

        valid_mask = tf.logical_or(pos_idx,neg_idx)
        gt_logits = tf.reshape(gt_logits,[-1])
        gt_logits = tf.boolean_mask(gt_logits,valid_mask)
        pred_logits = tf.boolean_mask(pred_logits,valid_mask)
        gt_anchor_deltas = tf.reshape(gt_anchors_deltas,[-1,box_dim])
        gt_anchor_deltas = tf.boolean_mask(gt_anchor_deltas,pos_idx)
        pred_anchor_deltas = tf.boolean_mask(pred_anchor_deltas,pos_idx)

        objectness_loss, localization_loss = ssd_losses(
            gt_logits,
            gt_anchor_deltas,
            pred_logits,
            pred_anchor_deltas,
        )
        normalizer = 1.0 / (batch_size* self.batch_size_per_image)
        loss_cls = objectness_loss * normalizer  # cls: classification loss
        loss_loc = localization_loss * normalizer  # loc: localization loss
        losses = {"loss_ssd_cls": loss_cls, "loss_ssd_loc": loss_loc}

        return losses

    @wmlt.add_name_scope
    def inference(self, inputs, box_cls, box_delta, anchors):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`RetinaNetHead.forward`
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(anchors[0].get_shape()) == 2, "error anchors dims"
        anchors_size = [tf.shape(x)[0] for x in anchors]
        anchors = tf.concat(anchors, axis=0)

        box_cls = [reshape_to_N_HWA_K(x, self.num_classes+1) for x in box_cls]
        box_delta = [reshape_to_N_HWA_K(x, 4) for x in box_delta]
        box_cls = tf.concat(box_cls, axis=1)
        box_delta = tf.concat(box_delta, axis=1)

        results = wmlt.static_or_dynamic_map_fn(
            lambda x: self.inference_single_image(x[0], x[1], anchors, anchors_size), elems=[box_cls, box_delta],
            dtype=(tf.float32, tf.int32, tf.float32, tf.int32),
            back_prop=False)
        outdata = {RD_BOXES: results[0], RD_LABELS: results[1], RD_PROBABILITY: results[2], RD_LENGTH: results[3]}
        if self.cfg.GLOBAL.SUMMARY_LEVEL <= SummaryLevel.DEBUG:
            wsummary.detection_image_summary(images=inputs[IMAGE],
                                             boxes=outdata[RD_BOXES], classes=outdata[RD_LABELS],
                                             scores=outdata[RD_PROBABILITY],
                                             lengths=outdata[RD_LENGTH],
                                             name="SSD_result",
                                             category_index=DataLoader.category_index)
        return outdata

    @wmlt.add_name_scope
    def inference_single_image(self, box_cls, box_delta, anchors, anchors_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls;[WxHxA(concat),K], logits, not probability
            box_delta [WxHxA(concat),box_dim]
            anchors [WxHxA(concat),box_dim]
            anchors_size: anchors'size per level
        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        total_num_classes = self.num_classes+1
        # Iterate over every feature level
        box_cls = tf.split(box_cls, num_or_size_splits=anchors_size)
        box_delta = tf.split(box_delta, num_or_size_splits=anchors_size)
        anchors = tf.split(anchors, num_or_size_splits=anchors_size)
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            box_cls_i = tf.nn.softmax(box_cls_i)
            box_cls_i = box_cls_i[:,1:] #remove background
            probs,classes_idxs = tf.nn.top_k(box_cls_i,k=1)
            probs = tf.reshape(probs,[-1])
            classes_idxs = tf.reshape(classes_idxs,[-1])
            classes_idxs = classes_idxs+1 #add the offset of background


            # Keep top k top scoring indices only.
            num_topk = tf.minimum(self.topk_candidates, tf.shape(box_reg_i)[0])
            predicted_prob, topk_idxs = tf.nn.top_k(probs, num_topk)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = tf.greater(predicted_prob, self.score_threshold)
            predicted_prob = tf.boolean_mask(predicted_prob, keep_idxs)
            topk_idxs = tf.boolean_mask(topk_idxs, keep_idxs)

            classes_idxs = tf.gather(classes_idxs,topk_idxs)
            # ssd use cls agnostic
            # after nms this will be fixed
            box_reg_i = tf.gather(box_reg_i, topk_idxs)
            anchors_i = tf.gather(anchors_i, topk_idxs)

            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            tf.concat(x, axis=0) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        x, y = wmlt.sort_data(key=scores_all, datas=[boxes_all, class_idxs_all])
        boxes_all, class_idxs_all = y
        scores_all, _ = x
        nms = functools.partial(tfop.boxes_nms, threshold=self.nms_threshold, classes_wise=True,
                                k=self.max_detections_per_image)
        boxes, labels, nms_idxs = nms(bboxes=boxes_all, classes=class_idxs_all)
        scores = tf.gather(scores_all, nms_idxs)

        candiate_nr = self.max_detections_per_image
        len = tf.shape(labels)[0]
        boxes = tf.pad(boxes, paddings=[[0, candiate_nr - len], [0, 0]])
        labels = tf.pad(labels, paddings=[[0, candiate_nr - len]])
        scores = tf.pad(scores, paddings=[[0, candiate_nr - len]])

        return [boxes, labels, scores, len]




