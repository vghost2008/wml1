#coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
import wnn
import functools
import wtfop.wtfop_ops as wop
import object_detection2.bboxes as odbox
from object_detection2.standard_names import *
import wmodule
from object_detection2.modeling.sampling import subsample_labels_by_negative_loss
from .onestage_tools import *



def ssd_losses(
    gt_logits,
    gt_anchor_deltas,
    pred_logits,
    pred_anchor_deltas,
):
    localization_loss = tf.losses.huber_loss(
        pred_anchor_deltas,
        gt_anchor_deltas,
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

    def inference(self, inputs,box_cls, box_delta,anchors):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`RetinaNetHead.forward`
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        batch_size = inputs[IMAGE].get_shape().as_list()[0]
        anchors_bach_size = anchors.get_shape().as_list()[0]

        results = []
        box_cls = [reshape_to_N_HWA_K(x,self.num_classes+1) for x in box_cls]
        box_delta = [reshape_to_N_HWA_K(x,4) for x in box_delta]

        for img_idx in range(batch_size):
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_delta]
            anchors_idx = 0 if anchors_bach_size<batch_size else img_idx
            anchors_per_image = anchors[anchors_idx]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, anchors_per_image,
            )
            results.append(results_per_image)
        results = list(zip(results))
        results = [tf.stack(x,axis=0) for x in results]
        outdata = {RD_BOXES:results[0],RD_LABELS:results[1],RD_PROBABILITY:results[2],RD_LENGTH:results[3]}
        return outdata

    def inference_single_image(self, box_cls, box_delta, anchors):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        total_num_classes = self.num_classes+1
        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            box_cls_i = tf.nn.softmax(tf.reshape(box_cls_i,[-1]))

            # Keep top k top scoring indices only.
            num_topk = tf.min(self.topk_candidates, tf.shape(box_reg_i)[0])
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = tf.nn.top_k(box_cls_i,num_topk,descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = tf.greater(predicted_prob,self.score_threshold)
            predicted_prob = tf.boolean_mask(predicted_prob,keep_idxs)
            topk_idxs = tf.boolean_mask(topk_idxs,keep_idxs)

            anchor_idxs = topk_idxs // total_num_classes
            classes_idxs = topk_idxs % total_num_classes

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
        boxes,labels,nms_idxs,lens = nms(bboxes=boxes_all,classes=class_idxs_all)
        scores = tf.gather(scores_all,nms_idxs)

        candiate_nr = self.max_detections_per_image
        len = tf.shape(labels)[0]
        boxes = tf.pad(boxes, paddings=[[0, candiate_nr - len], [0, 0]])
        labels = tf.pad(labels, paddings=[[0, candiate_nr - len]])
        scores = tf.pad(scores, paddings=[[0, candiate_nr - len]])

        return boxes,labels,scores,len


