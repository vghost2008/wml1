#For test purpose
import tensorflow as tf
import wmodule
from basic_tftools import channel
from .build import META_ARCH_REGISTRY
from object_detection2.modeling.build import build_outputs
from object_detection2.modeling.backbone.build import build_backbone
from object_detection2.modeling.anchor_generator import build_anchor_generator
from object_detection2.modeling.box_regression import CenterNet2Box2BoxTransform
from object_detection2.modeling.matcher import Matcher
from object_detection2.modeling.onestage_heads.build import build_onestage_head
from object_detection2.modeling.mot_heads.build import *
import math
from object_detection2.standard_names import *
from object_detection2.modeling.onestage_heads.retinanet_outputs import *
from .meta_arch import MetaArch
from object_detection2.datadef import *
import object_detection2.od_toolkit as odtk
import wnnlayer as wnnl
from functools import partial
from object_detection2.snpe_toolkit.snpe_engine import SNPEEngine

slim = tf.contrib.slim

__all__ = ["SNPE"]

@META_ARCH_REGISTRY.register()
class SNPE(MetaArch):
    """
    Implement: Objects as Points
    """

    def __init__(self, cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)

        # fmt: off
        self.num_classes              = cfg.MODEL.NUM_CLASSES
        self.k = cfg.MODEL.SNPE.K
        # fmt: on
        self.snpe = SNPEEngine(cfg.MODEL.SNPE.DLC_PATH,
                          output_names=['shared_head/ct_regr/Conv_1/BiasAdd','shared_head/heat_ct/Conv_1/BiasAdd',
                                        'shared_head/hw_regr/Conv_1/BiasAdd','shared_head/l2_normalize'],
                          output_layers=["shared_head/l2_normalize/Square", "shared_head/hw_regr/Conv_1/Conv2D",
                                         "shared_head/ct_regr/Conv_1/Conv2D","shared_head/heat_ct/Conv_1/Conv2D"],
                          output_shapes=[[1,135,240,2],[1,135,240,1],[1,135,240,2],[1,135,240,64]])
        # Matching and loss
        self.box2box_transform = CenterNet2Box2BoxTransform(num_classes=self.num_classes,k=self.k)

    def forward(self, batched_inputs):
        outputs = self.snpe.tf_forward(batched_inputs[IMAGE])

        head_outputs = {}
        head_outputs['offset'] = outputs[0]
        head_outputs['heatmaps_ct'] = outputs[1]
        head_outputs['hw'] = outputs[2]
        head_outputs['id_embedding'] = outputs[3]
        head_outputs = [head_outputs]

        gt_boxes = batched_inputs.get(GT_BOXES,None)
        gt_length = batched_inputs.get(GT_LENGTH,None)
        gt_labels = batched_inputs.get(GT_LABELS,None)

        outputs = build_outputs(name=self.cfg.MODEL.SNPE.OUTPUTS,
            cfg=self.cfg.MODEL.SNPE,
            parent=self,
            box2box_transform=self.box2box_transform,
            head_outputs = head_outputs,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_length=gt_length,
            max_detections_per_image=self.cfg.TEST.DETECTIONS_PER_IMAGE,
        )

        if self.is_training:
            if self.cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
                results = outputs.inference(inputs=batched_inputs,head_outputs=head_outputs)
            else:
                results = {}

            return results,outputs.losses()
        else:
            results = outputs.inference(inputs=batched_inputs,
                                        head_outputs=head_outputs)
            return results,{}