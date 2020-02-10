#coding=utf-8
import tensorflow as tf
import wmodule
import wml_tfutils as wmlt
from object_detection2.datadef import EncodedData
import wtfop.wtfop_ops as wop
import functools
import numpy as np

slim = tf.contrib.slim

class FastRCNNOutputs(wmodule.WChildModule):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self, cfg,parent,box2box_transform, pred_class_logits, pred_proposal_deltas,proposals:EncodedData,**kwargs
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        super().__init__(cfg,parent,**kwargs)
        self.box2box_transform = box2box_transform
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas

        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = proposals
        gt_logits_i = proposals.gt_object_logits
        '''
        gt_logits_i's shape is [batch_size,box_nr]
        '''
        self.gt_classes = tf.reshape(gt_logits_i,[-1])
        self.threshold = 0.05
        self.nms = None
        self.topk_per_image = -1

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        '''num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
            storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)'''
        pass

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        return tf.losses.sparse_softmax_cross_entropy(logits=self.pred_class_logits, labels=self.gt_classes,
                                               loss_collection=None,
                                               reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        #gt_anchor_deltas = self.box2box_transform.get_deltas(self.anchors,self.gt_boxes,gt_objectness_logits_i,indices)
        gt_proposal_deltas = self.box2box_transform.get_deltas_by_proposals_data(self.proposals)
        batch_size,box_nr,box_dim = wmlt.combined_static_and_dynamic_shape(gt_proposal_deltas)
        gt_proposal_deltas = tf.reshape(gt_proposal_deltas,[batch_size*box_nr,box_dim])
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.get_shape().as_list()[-1] == box_dim
        num_classes = self.pred_class_logits.get_shape().as_list()[-1]
        fg_num_classes = num_classes-1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = tf.greater(self.gt_classes,0)
        gt_proposal_deltas = tf.boolean_mask(gt_proposal_deltas,fg_inds)
        pred_proposal_deltas = tf.boolean_mask(self.pred_proposal_deltas,fg_inds)
        gt_logits_i = tf.boolean_mask(self.gt_classes,fg_inds)
        if not cls_agnostic_bbox_reg:
            pred_proposal_deltas = tf.reshape(pred_proposal_deltas,[-1,fg_num_classes,box_dim])
            pred_proposal_deltas = wmlt.select_2thdata_by_index_v2(pred_proposal_deltas, gt_logits_i- 1)

        loss_box_reg = tf.losses.huber_loss(
            predictions=pred_proposal_deltas, labels=gt_proposal_deltas,
            loss_collection=None,
            reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        )
        num_samples = wmlt.num_elements(self.gt_classes)
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg /num_samples
        return loss_box_reg

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
        }

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        boxes = self.box2box_transform.apply_deltas(deltas=self.pred_proposal_deltas,boxes=self.proposals.box)
        return boxes

    def predict_boxes_for_gt_classes(self):
        '''def get_predictionv2(class_prediction,
                             bboxes_regs,
                             proposal_bboxes,
                             limits=None,
                             prio_scaling=[0.1, 0.1, 0.2, 0.2],
                             threshold=0.5,
                             candiate_nr=1500,
                             classes_wise=False,
                             nms=None):'''
        proposal_bboxes = self.proposals.box
        gt_proposal_deltas = self.proposals.box.get_shape()[-1]
        batch_size,bor_nr,box_dim = gt_proposal_deltas.get_shape().as_list()
        classes_wise = (self.pred_proposal_deltas.get_shape().as_list()[-1] != box_dim)
        if proposal_bboxes.get_shape().as_list()[0] == 1:
            '''
            In single stage model, the proposal box are anchor boxes(or default boxes) and for any batch the anchor boxes is the same.
            '''
            proposal_bboxes = tf.squeeze(proposal_bboxes, axis=0)
            boxes, labels, probability, res_indices, lens = tf.map_fn(
                lambda x: self.__get_prediction(x[0], x[1], proposal_bboxes,
                                             self.threshold,
                                             classes_wise=classes_wise,
                                             nms=self.nms),
                elems=(self.pred_class_logits, self.pred_proposal_deltas), dtype=(tf.float32, tf.int32, tf.float32, tf.int32, tf.int32)
                )
        else:
            boxes, labels, probability, res_indices, lens = tf.map_fn(
                lambda x: self.__get_prediction(x[0], x[1], x[2],
                                             self.threshold,
                                             classes_wise=classes_wise,
                                             nms=self.nms),
                elems=(self.pred_class_logits, self.pred_proposal_deltas, self.proposals.box),
                dtype=(tf.float32, tf.int32, tf.float32, tf.int32, tf.int32)
                )

        results = {"boxes":boxes,"labels":labels,"probability":probability,"indices":res_indices,"len":lens}

        return results

    '''
    this version of get_prediction have no batch dim.

    class_prediction:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率
    shape为[X,num_classes]

    bboxes_regs:模型预测的每个proposal_bboxes/anchro boxes/default boxes到目标真实bbox的回归参数
    shape为[X,4](classes_wise=Flase)或者(X,num_classes,4](classes_wise=True)

    proposal_bboxes:候选box
    shape为[X,4]

    threshold:选择class_prediction的阀值

    nms_threshold: nms阀值

    candiate_nr:选择proposal_bboxes中预测最好的candiate_nr个bboxes进行筛选

    limits:[4,2],用于对回归参数的范围进行限制，分别对应于cy,cx,h,w的回归参数，limits的值对应于prio_scaling=[1,1,1,1]是的设置
    prio_scaling:在encode_boxes以及decode_boxes是使用的prio_scaling参数

    nms:nms函数,是否使用softnms,使用soft nms与不使用soft nms时, nms_threshold的意义有很大的区别， 不使用soft nms时，nms_threshold表示
    IOU小于nms_threshold的两个bbox为不同目标，使用soft nms时，nms_threshold表示得分高于nms_threshold的才是真目标

    返回:
    boxes:[candiate_nr,4]
    labels:[candiate_nr]
    probability:[candiate_nr]
    len: the available boxes number
    '''

    def __get_predictionv2(self,class_prediction,
                           bboxes_regs,
                           proposal_bboxes,
                           threshold=0.5,
                           classes_wise=False,
                           nms=None):
        # 删除背景
        class_prediction = class_prediction[:, 1:]
        probability, nb_labels = tf.nn.top_k(class_prediction, k=1)
        # 背景的类别为0，前面已经删除了背景，需要重新加上
        labels = nb_labels + 1
        ndims = class_prediction.get_shape().ndims
        probability = tf.squeeze(probability, axis=ndims - 1)
        labels = tf.squeeze(labels, axis=ndims - 1)
        res_indices = tf.range(tf.shape(labels)[0])

        # 按类别在bboxes_regs选择相应类的回归参数
        if classes_wise:
            nb_labels = tf.reshape(nb_labels, [-1])
            bboxes_regs = wmlt.select_2thdata_by_index(bboxes_regs, nb_labels)
        proposal_bboxes.get_shape().assert_is_compatible_with(bboxes_regs.get_shape())
        '''
        NMS前数据必须已经排好序
        通过top_k+gather排序
        '''
        probability, indices = tf.nn.top_k(probability, k=tf.shape(probability)[0])
        labels = wmlt.gather_in_axis(labels, indices, axis=0)
        bboxes_regs = wmlt.gather_in_axis(bboxes_regs, indices, axis=0)
        proposal_bboxes = wmlt.gather_in_axis(proposal_bboxes, indices, axis=0)
        res_indices = wmlt.gather_in_axis(res_indices, indices, axis=0)

        pmask = tf.greater(probability, threshold)
        probability = tf.boolean_mask(probability, pmask)
        labels = tf.boolean_mask(labels, pmask)
        proposal_bboxes = tf.boolean_mask(proposal_bboxes, pmask)
        boxes_regs = tf.boolean_mask(bboxes_regs, pmask)
        res_indices = tf.boolean_mask(res_indices, pmask)

        boxes = self.box2box_transform.apply_deltas(deltas=boxes_regs,boxes=proposal_bboxes)

        candiate_nr = tf.shape(probability)[0] if self.topk_per_image<0 else self.topk_per_image#最多可返回candiate_nr个box

        boxes, labels, indices = nms(boxes, labels, confidence=probability)
        probability = tf.gather(probability, indices)
        res_indices = tf.gather(res_indices, indices)

        probability = probability[:self.topk_per_image]
        boxes = boxes[:self.topk_per_image]
        labels = labels[:self.topk_per_image]
        probability = probability[:self.topk_per_image]
        res_indices = res_indices[:self.topk_per_image]
        len = tf.shape(probability)[0]
        boxes = tf.pad(boxes, paddings=[[0, candiate_nr - len], [0, 0]])
        labels = tf.pad(labels, paddings=[[0, candiate_nr - len]])
        probability = tf.pad(probability, paddings=[[0, candiate_nr - len]])
        res_indices = tf.pad(res_indices, paddings=[[0, candiate_nr - len]])
        boxes = tf.reshape(boxes, [candiate_nr, 4])
        labels = tf.reshape(labels, [candiate_nr])
        probability = tf.reshape(probability, [candiate_nr])
        res_indices = tf.reshape(res_indices, [candiate_nr])
        return boxes, labels, probability, res_indices, len

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = tf.nn.softmax(self.pred_class_logits, dim=-1)
        return probs

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        self.topk_per_image = topk_per_image
        self.threshold = score_thresh
        self.nms = functools.partial(wop.boxes_nms, threshold=nms_thresh, classes_wise=True)
        return self.predict_boxes_for_gt_classes()

class FastRCNNOutputLayers(wmodule.WChildModule):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, cfg,parent, num_classes, cls_agnostic_bbox_reg=False, box_dim=4,**kwargs):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of classes include background classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super().__init__(cfg,parent=parent,**kwargs)
        self.num_classes = num_classes
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.box_dim = box_dim

    def forward(self, x):
        if len(x.get_shape()) > 2:
            shape = wmlt.combined_static_and_dynamic_shape(x)
            x = tf.reshape(x,[shape[0],-1])
        scores = slim.fully_connected(x,self.num_classes,activation_fn=None,
                                      normalizer_fn=None,scope="cls_score")
        foreground_num_classes = self.num_classes-1
        num_bbox_reg_classes = 1 if self.cls_agnostic_bbox_reg else foreground_num_classes
        proposal_deltas = slim.fully_connected(x,self.box_dim*num_bbox_reg_classes,activation_fn=None,
                                      normalizer_fn=None,scope="bbox_pred")
        return scores, proposal_deltas