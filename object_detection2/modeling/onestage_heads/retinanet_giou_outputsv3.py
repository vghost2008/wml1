#coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
import wnn
import functools
import wtfop.wtfop_ops as wop
import object_detection2.bboxes as odbox
from object_detection2.standard_names import *
from .retinanet_outputs import RetinaNetOutputs
import wmodule
from object_detection2.modeling.onestage_heads.onestage_tools import *
from object_detection2.datadef import *
from object_detection2.config.config import global_cfg
import object_detection2.wlayers as odl
from object_detection2.modeling.build import HEAD_OUTPUTS
from .retinanet_outputs import RetinaNetOutputs
from object_detection2.modeling.matcher import Matcher
from object_detection2.data.dataloader import DataLoader
import wsummary


'''
Use GIOU loss instated the official huber loss for box regression.
'''
@HEAD_OUTPUTS.register()
class RetinaNetGIOUOutputsV3(RetinaNetOutputs):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.boxes_anchor_matcher = Matcher(
            thresholds=[0.3],
            allow_low_quality_matches=False,
            cfg=self.cfg,
            parent=self
        )
    def _get_regression_ground_truth(self):
        """
        Returns:
            gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """
        res = self.boxes_anchor_matcher(self.anchors,self.gt_boxes,self.gt_labels,self.gt_length,
                                  boxes_len = self.anchors_lens)

        gt_objectness_logits_i, scores, indices  = res

        gt_anchor_deltas = wmlt.batch_gather(self.gt_boxes,indices)
        return gt_objectness_logits_i, gt_anchor_deltas

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
    def regression_losses(self):
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

        assert len(self.pred_logits[0].get_shape()) == 4, "error logits dim"
        assert len(self.pred_anchor_deltas[0].get_shape()) == 4, "error anchors dim"

        gt_classes, gt_anchors_deltas = self._get_regression_ground_truth()
        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            self.pred_logits, self.pred_anchor_deltas, self.num_classes
        )  # Shapes: (N, R, K) and (N, R, 4), respectively.

        foreground_idxs = (gt_classes > 0)
        num_foreground = tf.reduce_sum(tf.cast(foreground_idxs, tf.int32))

        # regression loss
        pred_anchor_deltas = tf.boolean_mask(pred_anchor_deltas, foreground_idxs)
        gt_anchors_deltas = tf.boolean_mask(gt_anchors_deltas, foreground_idxs)
        B, X = wmlt.combined_static_and_dynamic_shape(foreground_idxs)
        anchors = tf.tile(self.anchors, [B, 1, 1])
        anchors = tf.boolean_mask(anchors, foreground_idxs)
        box = self.box2box_transform.apply_deltas(pred_anchor_deltas, anchors)
        reg_loss_sum = 1.0 - odl.giou(box, gt_anchors_deltas)
        loss_box_reg = tf.reduce_sum(reg_loss_sum) / tf.cast(tf.maximum(1, num_foreground), tf.float32)
        loss_box_reg = loss_box_reg * self.cfg.BOX_REG_LOSS_SCALE

        return loss_box_reg


    @wmlt.add_name_scope
    def cls_losses(self):
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

        loss_cls = loss_cls*self.cfg.BOX_CLS_LOSS_SCALE

        return loss_cls


    @wmlt.add_name_scope
    def losses(self):
        return {"loss_cls": self.cls_losses(), "loss_box_reg": self.regression_losses()}

