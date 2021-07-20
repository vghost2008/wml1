#coding=utf-8
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

slim = tf.contrib.slim

__all__ = ["CenterNet2"]

@META_ARCH_REGISTRY.register()
class CenterNet2(MetaArch):
    """
    Implement: Objects as Points
    """

    def __init__(self, cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)

        # fmt: off
        self.num_classes              = cfg.MODEL.NUM_CLASSES
        self.in_features              = cfg.MODEL.CENTERNET2.IN_FEATURES
        self.k = cfg.MODEL.CENTERNET2.K
        # fmt: on

        self.backbone = build_backbone(cfg,parent=self,*args,**kwargs)

        self.head = build_onestage_head(cfg.MODEL.CENTERNET2.HEAD_NAME,
                                        cfg=cfg.MODEL.CENTERNET2,
                                        parent=self,
                                        *args,**kwargs)

        # Matching and loss
        self.box2box_transform = CenterNet2Box2BoxTransform(num_classes=self.num_classes,k=self.k,
                                                            score_threshold=cfg.MODEL.CENTERNET2.SCORE_THRESH_TEST)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (H, W, C) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        batched_inputs = self.preprocess_image(batched_inputs)

        features = self.backbone(batched_inputs)
        if len(self.in_features) == 0:
            print(f"Error no input features for retinanet, use all features {features.keys()}")
            features = list(features.values())
        else:
            features = [features[f] for f in self.in_features]
        head_outputs = self.head(features)
        gt_boxes = batched_inputs.get(GT_BOXES,None)
        gt_length = batched_inputs.get(GT_LENGTH,None)
        gt_labels = batched_inputs.get(GT_LABELS,None)

        outputs = build_outputs(name=self.cfg.MODEL.CENTERNET2.OUTPUTS,
            cfg=self.cfg.MODEL.CENTERNET2,
            parent=self,
            box2box_transform=self.box2box_transform,
            head_outputs = head_outputs,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_length=gt_length,
            max_detections_per_image=self.cfg.TEST.DETECTIONS_PER_IMAGE,
        )

        if self.cfg.GLOBAL.SUMMARY_LEVEL <= SummaryLevel.DEBUG:
            for i, t_outputs in enumerate(head_outputs):
                wsummary.feature_map_summary(t_outputs['heatmaps_ct'], f'heatmaps_ct{i}')

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