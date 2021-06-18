#coding=utf-8
from object_detection2.modeling.meta_arch.retinanet import RetinaNet as _RetinaNet
from object_detection2.modeling.onestage_heads.fcos_outputs import FCOSGIOUOutputs as _FCOSGIOUOutputs
import math
import wml_tfutils as wmlt
import functools
from object_detection2.standard_names import *
from object_detection2.datadef import *
import tfop
from functools import partial
from object_detection2.modeling.build import HEAD_OUTPUTS

@HEAD_OUTPUTS.register()
class PGFCOSOutputs(_FCOSGIOUOutputs):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent=parent,*args,**kwargs)
        self.pre_nms_topk = {
            True: cfg.PRE_NMS_TOPK_TRAIN,
            False: cfg.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.POST_NMS_TOPK_TRAIN,
            False: cfg.POST_NMS_TOPK_TEST,
        }
        self.anchors_num_per_level = []
        self.topk_candidates = self.pre_nms_topk[self.is_training]

    def inference(self, inputs,box_cls, box_regression,center_ness):
        k = self.post_nms_topk[self.is_training]
        nms = partial(tfop.boxes_nms_nr2,k=k,
                      threshold=self.nms_threshold,
                      confidence=tf.convert_to_tensor([1.0],dtype=tf.float32))
        return super().inference(inputs,box_cls,box_regression,center_ness,nms=nms,pad=False)

    def losses(self):
        with tf.name_scope("fcospg_loss"):
            _loss = _FCOSGIOUOutputs.losses(self)
            if math.fabs(self.cfg.LOSS_SCALE-1.0)<1e-3:
                return _loss

            loss = {}
            for k,v in _loss.items():
                loss[k] = v*self.cfg.LOSS_SCALE

            return loss


