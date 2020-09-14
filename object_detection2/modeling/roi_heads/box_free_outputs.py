#coding=utf-8
import tensorflow as tf
import wmodule
import basic_tftools as btf
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

slim = tf.contrib.slim

@HEAD_OUTPUTS.register()
class BoxFreeOutputs(wmodule.WChildModule):
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
        self.pred_class_logits = pred_class_logits

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

    def softmax_cross_entropy_loss(self):
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
        return classes_loss*self.cfg.MODEL.ROI_HEADS.BOX_CLS_LOSS_SCALE

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        loss = {
            "fastrcnn_loss_cls": self.softmax_cross_entropy_loss(),
        }

        return loss

    def inference(self, score_thresh, 
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
            if scores is None:
                probability = tf.nn.softmax(self.pred_class_logits)
            else:
                probability = scores

            probability = probability[...,1:] #删除背景
            probability,labels = tf.nn.top_k(probability,k=1)
            probability = tf.squeeze(probability,axis=-1)
            labels = tf.squeeze(labels,axis=-1)+1  #加回删除的背景
            size = btf.combined_static_and_dynamic_shape(probability)[0]
            res_indices = tf.range(size)
            mask = tf.greater(probability,score_thresh)
            length = tf.reduce_sum(tf.cast(mask,tf.int32),axis=-1,keepdims=False)
            probability = tf.boolean_mask(probability,mask)
            boxes = tf.boolean_mask(proposal_boxes,mask)
            labels = tf.boolean_mask(labels,mask)
            res_indices = tf.boolean_mask(res_indices,mask)

            probability, indices= tf.nn.top_k(probability, k=tf.shape(probability)[0])
            labels = tf.expand_dims(tf.gather(labels, indices),axis=0)
            boxes = tf.expand_dims(tf.gather(boxes, indices),axis=0)
            res_indices = tf.expand_dims(tf.gather(res_indices, indices),axis=0)
            probability = tf.expand_dims(probability,axis=0)
            
            return {RD_PROBABILITY:probability,RD_BOXES:boxes,RD_LABELS:labels,RD_LENGTH:length,RD_INDICES:res_indices}
