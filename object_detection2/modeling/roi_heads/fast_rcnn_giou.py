#coding=utf-8
import tensorflow as tf
import wmodule
import wml_tfutils as wmlt
from object_detection2.datadef import EncodedData
from .build import HEAD_OUTPUTS
from .fast_rcnn import FastRCNNOutputs as _FastRCNNOutputs
import wtfop.wtfop_ops as wop
import functools
from object_detection2.datadef import *
import numpy as np
import wnn
import wsummary
import wml_tfutils as wmlt
import object_detection2.wlayers as odl

slim = tf.contrib.slim

@HEAD_OUTPUTS.register()
class FastRCNNGIOUOutputs(_FastRCNNOutputs):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self,*args,**kwargs
    ):
        super().__init__(*args,**kwargs)

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        with tf.name_scope("box_regression_loss"):
            gt_proposal_deltas = wmlt.batch_gather(self.proposals.gt_boxes,self.proposals.indices)
            batch_size,box_nr,box_dim = wmlt.combined_static_and_dynamic_shape(gt_proposal_deltas)
            gt_proposal_deltas = tf.reshape(gt_proposal_deltas,[batch_size*box_nr,box_dim])
            proposal_bboxes = tf.reshape(self.proposals.boxes,[batch_size*box_nr,box_dim])
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
            proposal_bboxes = tf.boolean_mask(proposal_bboxes,fg_inds)
            gt_logits_i = tf.boolean_mask(self.gt_classes,fg_inds)
            if not cls_agnostic_bbox_reg:
                pred_proposal_deltas = tf.reshape(pred_proposal_deltas,[-1,fg_num_classes,box_dim])
                pred_proposal_deltas = wmlt.select_2thdata_by_index_v2(pred_proposal_deltas, gt_logits_i- 1)

            pred_bboxes = self.box2box_transform.apply_deltas(pred_proposal_deltas,boxes=proposal_bboxes)
            loss_box_reg = odl.giou_loss(pred_bboxes, gt_proposal_deltas)
            loss_box_reg = tf.reduce_sum(loss_box_reg)
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

        return loss_box_reg

