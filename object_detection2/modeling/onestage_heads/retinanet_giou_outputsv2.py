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
from .retinanet_giou_outputs import RetinaNetGIOUOutputs
import wsummary
import numpy as np
from iotoolkit.coco_toolkit import ID_TO_TEXT

freq = {
"person": 30.52,
"car": 5.10,
"chair": 4.48,
"book": 2.87,
"bottle": 2.83,
"cup": 2.40,
"dining table": 1.83,
"bowl": 1.67,
"traffic light": 1.50,
"handbag": 1.44,
"umbrella": 1.33,
"bird": 1.26,
"boat": 1.25,
"truck": 1.16,
"bench": 1.14,
"sheep": 1.11,
"banana": 1.10,
"kite": 1.06,
"motorcycle": 1.01,
"backpack": 1.01,
"potted plant": 1.01,
"cow": 0.95,
"wine glass": 0.92,
"carrot": 0.91,
"knife": 0.90,
"broccoli": 0.85,
"donut": 0.83,
"bicycle": 0.83,
"skis": 0.77,
"vase": 0.77,
"horse": 0.77,
"tie": 0.76,
"cell phone": 0.75,
"orange": 0.74,
"cake": 0.74,
"sports ball": 0.74,
"clock": 0.74,
"suitcase": 0.72,
"spoon": 0.72,
"surfboard": 0.71,
"bus": 0.71,
"apple": 0.68,
"pizza": 0.68,
"tv": 0.68,
"couch": 0.67,
"remote": 0.66,
"sink": 0.65,
"skateboard": 0.64,
"elephant": 0.64,
"dog": 0.64,
"fork": 0.64,
"zebra": 0.62,
"airplane": 0.60,
"giraffe": 0.60,
"laptop": 0.58,
"tennis racket": 0.56,
"teddy bear": 0.56,
"cat": 0.55,
"train": 0.53,
"sandwich": 0.51,
"bed": 0.49,
"toilet": 0.48,
"baseball glove": 0.44,
"oven": 0.39,
"baseball bat": 0.38,
"hot dog": 0.34,
"keyboard": 0.33,
"snowboard": 0.31,
"frisbee": 0.31,
"refrigerator": 0.31,
"mouse": 0.26,
"stop sign": 0.23,
"toothbrush": 0.23,
"fire hydrant": 0.22,
"microwave": 0.19,
"scissors": 0.17,
"bear": 0.15,
"parking meter": 0.15,
"toaster": 0.03,
"hair drier": 0.02
}

'''
Use GIOU loss instated the official huber loss for box regression.
    '''
@HEAD_OUTPUTS.register()
class RetinaNetGIOUOutputsV2(RetinaNetGIOUOutputs):
    def __init__(
                self,
                *args,
        **kwargs,
    ):
        super().__init__(*args,**kwargs)
        self.weights = self.init_weights()
    
    def init_weights(self):
        w = np.ones(shape=[91],dtype=np.float32)
        for i in range(1,91):
            if i in ID_TO_TEXT:
                name = ID_TO_TEXT[i]['name']
                if name not in freq:
                    print(f"Error id {i}/{name} not in freq")
                    raise ValueError(name)
                else:
                    v = freq[name]
                    w[i] = v
        
        return w


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
        ref_weights = tf.gather(params=self.weights,indices=gt_classes_target)
        ref_weights = tf.expand_dims(ref_weights,axis=-1)
        ref_weights = tf.tile(ref_weights,[1,self.num_classes])
        N,_ = wmlt.combined_static_and_dynamic_shape(ref_weights)
        act_weights = tf.tile(tf.expand_dims(self.weights,axis=0),[N,1])[...,1:]
        weights = act_weights/ref_weights
        weights = tf.minimum(weights,1.0)
        gt_classes_target = tf.one_hot(gt_classes_target,depth=self.num_classes+1)
        gt_classes_target = gt_classes_target[:,1:]#RetinaNetGIou中没有背景, 因为背景index=0, 所以要在one hot 后去掉背景
        pred_class_logits = tf.boolean_mask(pred_class_logits,valid_idxs)

        # logits loss
        loss_cls = tf.reduce_sum(wnn.sigmoid_cross_entropy_with_logits_FL(
            labels = gt_classes_target,
            logits = pred_class_logits,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            weights=weights,
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
        loss_cls = loss_cls*self.cfg.BOX_CLS_LOSS_SCALE
        loss_box_reg = loss_box_reg*self.cfg.BOX_REG_LOSS_SCALE

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

