#coding=utf-8
import tensorflow as tf
import wmodule
import wml_tfutils as wmlt
from object_detection2.datadef import EncodedData
import wtfop.wtfop_ops as wop
import functools
from object_detection2.datadef import *
import numpy as np
import wnn
import wsummary
from .build import HEAD_OUTPUTS
import object_detection2.wlayers as odl
from object_detection2.modeling.matcher import Matcher
import object_detection2.bboxes as odb

slim = tf.contrib.slim

@HEAD_OUTPUTS.register()
class FastRCNNOutputs(wmodule.WChildModule):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self, cfg,parent,box2box_transform, pred_class_logits, pred_proposal_deltas,proposals:EncodedData,
            pred_iou_logits=None,
            **kwargs
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
            proposals: When training it's EncodedData, when inference, it's ProposalsData
        """
        super().__init__(cfg,parent,**kwargs)
        self.box2box_transform = box2box_transform
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas

        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = proposals
        self.pred_iou_logits = pred_iou_logits
        if self.is_training:
            gt_logits_i = proposals.gt_object_logits
            '''
            gt_logits_i's shape is [batch_size,box_nr]
            '''
            self.gt_classes = tf.reshape(gt_logits_i,[-1])

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        accuracy = wnn.accuracy_ratio(logits=self.pred_class_logits,labels=self.gt_classes)
        tf.summary.scalar("fast_rcnn/accuracy",accuracy)

    '''def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        wsummary.variable_summaries_v2(self.gt_classes,"gt_classes")
        wsummary.variable_summaries_v2(self.pred_class_logits,"pred_class_logits")
        if self.cfg.MODEL.ROI_HEADS.POS_LABELS_THRESHOLD>1e-3:
            with tf.name_scope("modify_gtclasses"):
                threshold = self.cfg.MODEL.ROI_HEADS.POS_LABELS_THRESHOLD
                scores = tf.reshape(self.proposals[ED_SCORES],[-1])
                gt_classes = self.gt_classes
                gt_classes = tf.where(tf.greater(scores,threshold),gt_classes,tf.zeros_like(gt_classes))
            classes_loss = tf.losses.sparse_softmax_cross_entropy(logits=self.pred_class_logits, labels=gt_classes,
                                               loss_collection=None,
                                               reduction=tf.losses.Reduction.MEAN)
        else:
            classes_loss = tf.losses.sparse_softmax_cross_entropy(logits=self.pred_class_logits, labels=self.gt_classes,
                                                                  loss_collection=None,
                                                                  reduction=tf.losses.Reduction.MEAN)

        wsummary.histogram_or_scalar(classes_loss,"fast_rcnn/classes_loss")
        return classes_loss*self.cfg.MODEL.ROI_HEADS.BOX_CLS_LOSS_SCALE'''
    
    def softmax_cross_entropy_loss(self):
        self._log_accuracy()
        wsummary.variable_summaries_v2(self.gt_classes,"gt_classes")
        wsummary.variable_summaries_v2(self.pred_class_logits,"pred_class_logits")
        scores = tf.stop_gradient(tf.reshape(self.proposals[ED_SCORES], [-1]))
        #weights = tf.abs(scores-0.5)*4
        weights = tf.minimum(tf.pow(tf.abs(scores-0.5),2)*100,1.0)
        weights = tf.stop_gradient(weights)
        wsummary.histogram_or_scalar(weights,"cls_loss_weights")
        if self.cfg.MODEL.ROI_HEADS.POS_LABELS_THRESHOLD>1e-3:
            with tf.name_scope("modify_gtclasses"):
                threshold = self.cfg.MODEL.ROI_HEADS.POS_LABELS_THRESHOLD
                gt_classes = self.gt_classes
                gt_classes = tf.where(tf.greater(scores,threshold),gt_classes,tf.zeros_like(gt_classes))
            classes_loss = tf.losses.sparse_softmax_cross_entropy(logits=self.pred_class_logits, labels=gt_classes,
                                               loss_collection=None,
                                               reduction=tf.losses.Reduction.NONE)
        else:
            classes_loss = tf.losses.sparse_softmax_cross_entropy(logits=self.pred_class_logits, labels=self.gt_classes,
                                                                  loss_collection=None,
                                                                  reduction=tf.losses.Reduction.NONE)

        classes_loss = weights*classes_loss
        classes_loss = tf.reduce_mean(classes_loss)
        wsummary.histogram_or_scalar(classes_loss,"fast_rcnn/classes_loss")
        return classes_loss*self.cfg.MODEL.ROI_HEADS.BOX_CLS_LOSS_SCALE

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        #gt_anchor_deltas = self.box2box_transform.get_deltas(self.anchors,self.gt_boxes,gt_objectness_logits_i,indices)
        with tf.name_scope("box_regression_loss"):
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
            # arg to smooth_l1_loss is False (otherwise it uses mean internally
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
                reduction=tf.losses.Reduction.SUM,
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

        wsummary.histogram_or_scalar(loss_box_reg,"fast_rcnn/box_reg_loss")

        return loss_box_reg*self.cfg.MODEL.ROI_HEADS.BOX_REG_LOSS_SCALE

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        loss = {
            "fastrcnn_loss_cls": self.softmax_cross_entropy_loss(),
            "fastrcnn_loss_box_reg": self.smooth_l1_loss(),
        }
        if self.pred_iou_logits is not None:
            loss['fastrcnn_loss_iou'] = self.get_pred_iou_loss()

        return loss

    def get_pred_iou_loss(self):
        if self.cfg.MODEL.ROI_HEADS.PRED_IOU_VERSION == 1:
            return self.get_pred_iou_lossv1()
        elif self.cfg.MODEL.ROI_HEADS.PRED_IOU_VERSION == 3:
            return self.get_pred_centerness_loss()
        elif self.cfg.MODEL.ROI_HEADS.PRED_IOU_VERSION == 4:
            return self.get_pred_iou_lossv0(threshold=0.4)
        else:
            return self.get_pred_iou_lossv0()

    def get_pred_iou_lossv0(self,threshold=None):
        '''
        使用proposals bboxes与gtbboxes的iou作为目标
        :return: 
        '''
        with tf.name_scope("get_pred_iouv0_loss"):
            gt_scores = self.proposals[ED_SCORES]
            gt_scores = tf.stop_gradient(tf.reshape(gt_scores,[-1]))
            pred_iou_logits = self.pred_iou_logits
            pred_iou_logits = tf.reshape(pred_iou_logits,[-1])
            if threshold is not None:
                pred_score = tf.nn.sigmoid(pred_iou_logits)
                mask0 = tf.greater(gt_scores,threshold)
                mask1 = tf.logical_and(tf.logical_not(mask0),tf.greater_equal(pred_score,threshold+0.05))
                mask = tf.logical_or(mask0,mask1)
                gt_scores = tf.boolean_mask(gt_scores,mask)
                pred_iou_logits = tf.boolean_mask(pred_iou_logits,mask)

            wsummary.histogram_or_scalar(gt_scores,"gt_iou_by_matcher")
            loss = wnn.sigmoid_cross_entropy_with_logits_FL(labels=gt_scores,
                                                            logits=pred_iou_logits)
            return tf.reduce_mean(loss)

    def get_pred_iou_lossv1(self):
        '''
        使用预测的bboxes与gtbboxes的iou作为目标
        :return: 
        '''
        with tf.name_scope("get_pred_iouv1_loss"):
            gt_proposal_deltas = wmlt.batch_gather(self.proposals.gt_boxes, tf.nn.relu(self.proposals.indices))
            batch_size, box_nr, box_dim = wmlt.combined_static_and_dynamic_shape(gt_proposal_deltas)
            gt_proposal_deltas = tf.reshape(gt_proposal_deltas, [batch_size * box_nr, box_dim])
            proposal_bboxes = tf.reshape(self.proposals.boxes, [batch_size * box_nr, box_dim])
            cls_agnostic_bbox_reg = self.pred_proposal_deltas.get_shape().as_list()[-1] == box_dim
            num_classes = self.pred_class_logits.get_shape().as_list()[-1]
            fg_num_classes = num_classes - 1
            pred_iou_logits = self.pred_iou_logits

            fg_inds = tf.greater(self.gt_classes, 0)
            gt_proposal_deltas = tf.boolean_mask(gt_proposal_deltas, fg_inds)
            pred_proposal_deltas = tf.boolean_mask(self.pred_proposal_deltas, fg_inds)
            proposal_bboxes = tf.boolean_mask(proposal_bboxes, fg_inds)
            gt_logits_i = tf.boolean_mask(self.gt_classes, fg_inds)
            pred_iou_logits_pos = tf.reshape(tf.boolean_mask(pred_iou_logits, fg_inds),[-1])
            pred_iou_logits_neg = tf.reshape(tf.boolean_mask(pred_iou_logits, tf.logical_not(fg_inds)),[-1])
            if not cls_agnostic_bbox_reg:
                pred_proposal_deltas = tf.reshape(pred_proposal_deltas, [-1, fg_num_classes, box_dim])
                pred_proposal_deltas = wmlt.select_2thdata_by_index_v2(pred_proposal_deltas, gt_logits_i - 1)

            pred_bboxes = self.box2box_transform.apply_deltas(pred_proposal_deltas, boxes=proposal_bboxes)
            loss_box_reg = odl.giou(pred_bboxes, gt_proposal_deltas)
            loss_box_reg = tf.stop_gradient(loss_box_reg)
            loss_pos = wnn.sigmoid_cross_entropy_with_logits_FL(labels=loss_box_reg,
                                                            logits=pred_iou_logits_pos)
            loss_pos = tf.reduce_mean(loss_pos)
            loss_neg = wnn.sigmoid_cross_entropy_with_logits_FL(labels=tf.zeros_like(pred_iou_logits_neg),
                                                            logits=pred_iou_logits_neg)
            loss_neg = tf.reduce_mean(loss_neg)*0.5
            tf.summary.scalar("iou_pos_loss",loss_pos)
            tf.summary.scalar("iou_neg_loss",loss_neg)

            loss = loss_pos+loss_neg

        return loss

    def get_pred_centerness_loss(self):
        with tf.name_scope("get_pred_centerness_loss"):
            gt_proposal_deltas = wmlt.batch_gather(self.proposals.gt_boxes, tf.nn.relu(self.proposals.indices))
            batch_size, box_nr, box_dim = wmlt.combined_static_and_dynamic_shape(gt_proposal_deltas)
            gt_proposal_deltas = tf.reshape(gt_proposal_deltas, [batch_size * box_nr, box_dim])
            proposal_bboxes = tf.reshape(self.proposals.boxes, [batch_size * box_nr, box_dim])
            cls_agnostic_bbox_reg = self.pred_proposal_deltas.get_shape().as_list()[-1] == box_dim
            num_classes = self.pred_class_logits.get_shape().as_list()[-1]
            fg_num_classes = num_classes - 1
            pred_iou_logits = self.pred_iou_logits

            fg_inds = tf.greater(self.gt_classes, 0)
            gt_proposal_deltas = tf.boolean_mask(gt_proposal_deltas, fg_inds)
            pred_proposal_deltas = tf.boolean_mask(self.pred_proposal_deltas, fg_inds)
            proposal_bboxes = tf.boolean_mask(proposal_bboxes, fg_inds)
            gt_logits_i = tf.boolean_mask(self.gt_classes, fg_inds)
            pred_iou_logits_pos = tf.reshape(tf.boolean_mask(pred_iou_logits, fg_inds),[-1])
            if not cls_agnostic_bbox_reg:
                pred_proposal_deltas = tf.reshape(pred_proposal_deltas, [-1, fg_num_classes, box_dim])
                pred_proposal_deltas = wmlt.select_2thdata_by_index_v2(pred_proposal_deltas, gt_logits_i - 1)

            pred_bboxes = self.box2box_transform.apply_deltas(pred_proposal_deltas, boxes=proposal_bboxes)
            pred_bboxes = odb.to_cxyhw(proposal_bboxes)
            gt_bboxes = odb.to_cxyhw(gt_proposal_deltas)
            deltas = tf.abs(gt_bboxes[...,:2]-pred_bboxes[...,:2])*2
            wsummary.histogram_or_scalar(deltas,"centerness_deltas")
            centerness = 1-tf.reduce_max(deltas/(gt_bboxes[...,2:]+1e-8),axis=-1,keepdims=False)
            wsummary.histogram_or_scalar(centerness,"centerness")
            loss_pos = wnn.sigmoid_cross_entropy_with_logits_FL(labels=centerness,
                                                                logits=pred_iou_logits_pos)
            wsummary.histogram_or_scalar(tf.nn.sigmoid(pred_iou_logits_pos),"pred_centerness")
            loss_pos = tf.reduce_mean(loss_pos)

            tf.summary.scalar("centerness_loss",loss_pos)

            loss = loss_pos

        return loss

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if self.is_training:
            proposals_boxes = self.proposals.boxes
        else:
            proposals_boxes = self.proposals[PD_BOXES]
        batch_size,box_nr,box_dim = wmlt.combined_static_and_dynamic_shape(proposals_boxes)
        pred_proposal_deltas = tf.reshape(self.pred_proposal_deltas,[batch_size,box_nr,box_dim])
        boxes = self.box2box_transform.apply_deltas(deltas=pred_proposal_deltas,boxes=proposals_boxes)
        return boxes

    def predict_boxes_for_gt_classes(self):
        '''
        当后继还有mask或keypoint之类分支，它们可以在与RCNN相同的输入(即RPN的输出上处理), 也可以在RCNN的输出上处理，
        这个函数用于辅助完成在RCNN的输出结果上进行处理的功能，现在的输入的proposal box已经是[batch_size,N,4], 经过处理后
		还是这个形状
		Detectron2所有的配置都没有使用这一功能，但理论上来说这样更好（但训练的效率更低）
		为了防止前期不能生成好的结果，这里实现相对于Detectron2来说加入了gt_boxes
        :return:
        [batch_size,box_nr,box_dim]
        '''
        with tf.name_scope("predict_boxes_for_gt_classes"):
            predicted_boxes = self.predict_boxes()
            B = self.proposals[PD_BOXES].get_shape().as_list()[-1]
            # If the box head is class-agnostic, then the method is equivalent to `predicted_boxes`.
            if predicted_boxes.get_shape().as_list()[-1] > B:
                gt_classes = tf.reshape(self.proposals.gt_object_logits,[-1])
                batch_size,box_nr,box_dim = wmlt.combined_static_and_dynamic_shape(self.proposals[PD_BOXES])
                predicted_boxes = tf.reshape(predicted_boxes,[batch_size*box_nr,-1,box_dim])
                predicted_boxes = wmlt.batch_gather(predicted_boxes,gt_classes)
                predicted_boxes = tf.reshape(predicted_boxes,[batch_size,box_nr,box_dim])
            return predicted_boxes

    '''
    this version of get_prediction have no batch dim.

    class_prediction:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率, 不能使用logits
    因为涉及到使用阀值过滤
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
    indices:[candiate_nr], 输出box所对应的输入box序号
    len: the available boxes number
    '''

    def prediction_on_single_image(self,class_prediction,
                           bboxes_regs,
                           proposal_bboxes,
                           threshold=0.5,
                           classes_wise=False,
                           topk_per_image=-1,
                           ious=None,
                           nms=None):
        with tf.name_scope("prediction_on_single_image"):
            # 删除背景
            class_prediction = class_prediction[:, 1:]
            num_classes = class_prediction.get_shape().as_list()[-1]
            probability, nb_labels = tf.nn.top_k(class_prediction, k=1)
            # 背景的类别为0，前面已经删除了背景，需要重新加上
            labels = nb_labels + 1

            '''num_classes = class_prediction.get_shape().as_list()[-1]-1
            probability, labels = tf.nn.top_k(class_prediction, k=1)
            probability = tf.where(tf.greater(labels,0),probability,tf.zeros_like(probability))
            nb_labels = tf.where(tf.greater(labels,0),labels-1,labels)'''

            ndims = class_prediction.get_shape().ndims
            probability = tf.squeeze(probability, axis=ndims - 1)
            labels = tf.squeeze(labels, axis=ndims - 1)
            res_indices = tf.range(tf.shape(labels)[0])

            # 按类别在bboxes_regs选择相应类的回归参数
            if classes_wise:
                nb_labels = tf.reshape(nb_labels, [-1])
                box_nr,box_dim = wmlt.combined_static_and_dynamic_shape(proposal_bboxes)
                bboxes_regs = tf.reshape(bboxes_regs,[box_nr,num_classes,box_dim])
                bboxes_regs = wmlt.batch_gather(bboxes_regs,nb_labels)
            del nb_labels
            proposal_bboxes.get_shape().assert_is_compatible_with(bboxes_regs.get_shape())
            '''
            NMS前数据必须已经排好序
            通过top_k+gather排序
            '''
            if ious is None:
                probability, indices = tf.nn.top_k(probability, k=tf.shape(probability)[0])
            else:
                _, indices = tf.nn.top_k(ious, k=tf.shape(ious)[0])
                probability = tf.gather(probability,indices)

            labels = tf.gather(labels, indices)
            bboxes_regs = tf.gather(bboxes_regs, indices)
            proposal_bboxes = tf.gather(proposal_bboxes, indices)
            res_indices = tf.gather(res_indices, indices)

            pmask = tf.greater(probability, threshold)
            probability = tf.boolean_mask(probability, pmask)
            labels = tf.boolean_mask(labels, pmask)
            proposal_bboxes = tf.boolean_mask(proposal_bboxes, pmask)
            boxes_regs = tf.boolean_mask(bboxes_regs, pmask)
            res_indices = tf.boolean_mask(res_indices, pmask)

            boxes = self.box2box_transform.apply_deltas(deltas=boxes_regs,boxes=proposal_bboxes)

            candiate_nr = tf.shape(probability)[0] if topk_per_image<0 else topk_per_image#最多可返回candiate_nr个box

            if ious is None:
                boxes, labels, indices = nms(boxes, labels, confidence=probability)
            else:
                ious = tf.gather(ious,res_indices)
                boxes, labels, indices = nms(boxes, labels, confidence=ious)
            probability = tf.gather(probability, indices)
            res_indices = tf.gather(res_indices, indices)

            probability = probability[:topk_per_image]
            boxes = boxes[:topk_per_image]
            labels = labels[:topk_per_image]
            probability = probability[:topk_per_image]
            res_indices = res_indices[:topk_per_image]
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

    def inference(self, score_thresh, nms_thresh, topk_per_image,pred_iou_logits=None,
                  proposal_boxes=None,scores=None):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
            scores:[batch_size,box_nr,num_classes+1]
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        with tf.name_scope("fast_rcnn_outputs_inference"):
            output_fix_nr = self.cfg.MODEL.ROI_HEADS.OUTPUTS_FIX_NR_BOXES
            if output_fix_nr < 1:
                nms = functools.partial(wop.boxes_nms, threshold=nms_thresh,
                                        classes_wise=self.cfg.MODEL.ROI_HEADS.CLASSES_WISE_NMS)
            else:
                nms = functools.partial(wop.boxes_nms_nr2, threshold=nms_thresh,
                                        classes_wise = self.cfg.MODEL.ROI_HEADS.CLASSES_WISE_NMS,
                                        k=output_fix_nr,
                                        allow_less_output=True)

            if proposal_boxes is None:
                proposal_boxes = self.proposals[PD_BOXES]
            batch_size,bor_nr,box_dim = proposal_boxes.get_shape().as_list()
            _,L = wmlt.combined_static_and_dynamic_shape(self.pred_proposal_deltas)
            if scores is None:
                probability = tf.nn.softmax(self.pred_class_logits)
            else:
                probability = scores
            if pred_iou_logits is not None:
                ious = tf.nn.sigmoid(pred_iou_logits,"pred_iou")

                if self.cfg.GLOBAL.SUMMARY_LEVEL <= SummaryLevel.RESEARCH and not self.is_training:
                    matcher = Matcher(
                        [1e-3],
                        allow_low_quality_matches=False,
                        cfg=self.cfg,
                        parent=self
                    )
                    mh_res0 = matcher(proposal_boxes,
                                  self.parent.batched_inputs[GT_BOXES],
                                  self.parent.batched_inputs[GT_LABELS],
                                  self.parent.batched_inputs[GT_LENGTH])
                    add_to_research_datas("pb_scores",mh_res0[1],[-1]) #proposal_boxes与gt boxes的 iou
                    add_to_research_datas("pb_ious",ious,[-1]) #预测的iou
                    assert len(probability.get_shape())==2, "error probability shape"

                raw_probability = tf.expand_dims(probability,axis=0) #only support batch_size=1
                #probability = ious*probability

            else:
                if self.cfg.GLOBAL.SUMMARY_LEVEL <= SummaryLevel.RESEARCH and not self.is_training:
                    matcher = Matcher(
                        [1e-3],
                        allow_low_quality_matches=False,
                        cfg=self.cfg,
                        parent=self
                    )
                    mh_res0 = matcher(proposal_boxes,
                                      self.parent.batched_inputs[GT_BOXES],
                                      self.parent.batched_inputs[GT_LABELS],
                                      self.parent.batched_inputs[GT_LENGTH])
                    add_to_research_datas("pb_scores",mh_res0[1],[-1]) #proposal_boxes与gt boxes的 iou
                    assert len(probability.get_shape())==2, "error probability shape"

            total_box_nr,K = wmlt.combined_static_and_dynamic_shape(probability)
            probability = tf.reshape(probability,[batch_size,-1,K])
            if pred_iou_logits is not None:
                if self.cfg.MODEL.ROI_HEADS.PRED_IOU_VERSION == 3:
                    ious = tf.reshape(ious,[batch_size,-1,1])
                    probability = probability*ious
                    ious = None
                    print("Pred centerness.")
                else:
                    ious = tf.reshape(ious,[batch_size,-1])

            pred_proposal_deltas = tf.reshape(self.pred_proposal_deltas,[batch_size,-1,L])
            classes_wise = (L != box_dim)

            if pred_iou_logits is None or ious is None or (not self.cfg.MODEL.ROI_HEADS.USE_IOU_IN_TEST):
                boxes, labels, probability, res_indices, lens = tf.map_fn(
                    lambda x: self.prediction_on_single_image(x[0], x[1], x[2],
                                                    score_thresh,
                                                    classes_wise=classes_wise,
                                                    topk_per_image=topk_per_image,
                                                    nms=nms),
                    elems=(probability, pred_proposal_deltas, proposal_boxes),
                    dtype=(tf.float32, tf.int32, tf.float32, tf.int32, tf.int32)
                )
            else:
                #nms时使用iou排序
                boxes, labels, probability, res_indices, lens = tf.map_fn(
                    lambda x: self.prediction_on_single_image(x[0], x[1], x[2],
                                                              score_thresh,
                                                              classes_wise=classes_wise,
                                                              topk_per_image=topk_per_image,
                                                              ious=x[3],
                                                              nms=nms),
                    elems=(probability, pred_proposal_deltas, proposal_boxes,ious),
                    dtype=(tf.float32, tf.int32, tf.float32, tf.int32, tf.int32)
                )

        with tf.name_scope("remove_null_boxes"):
            #max_len=0会引导程序异常退出，原因未知
            max_len = tf.maximum(1,tf.reduce_max(lens))
            boxes = boxes[:,:max_len]
            labels = labels[:,:max_len]
            probability = probability[:,:max_len]
            res_indices = res_indices[:,:max_len]

        results = {RD_BOXES:boxes,RD_LABELS:labels,RD_PROBABILITY:probability,RD_INDICES:res_indices,RD_LENGTH:lens}
        if self.cfg.GLOBAL.SUMMARY_LEVEL <= SummaryLevel.RESEARCH and not self.is_training:
            with tf.device("/cpu:0"):
                if pred_iou_logits is not None and ious is not None:
                    ious = wmlt.batch_gather(ious,res_indices)
                else:
                    ious = None

                scores0 = wmlt.batch_gather(mh_res0[1],res_indices)
                l = lens[0]
                mh_res1 = matcher(boxes[:,:l],
                                 self.parent.batched_inputs[GT_BOXES],
                                 self.parent.batched_inputs[GT_LABELS],
                                 self.parent.batched_inputs[GT_LENGTH])
                add_to_research_datas("rd_scores", mh_res1[1][:,:l],[-1])

                if ious is not None:
                    add_to_research_datas("rd_ious", ious[:,:l],[-1])

                if pred_iou_logits is not None:
                    probs = wmlt.batch_gather(raw_probability,res_indices)
                    add_to_research_datas("rd_probs", probs[:,:l],[-1])
                else:
                    add_to_research_datas("rd_probs", probability[:,:l],[-1])
                add_to_research_datas("rd_probs", probability[:,:l],[-1])
                add_to_research_datas("rd_scores_old", scores0[:,:l],[-1])

        return results
