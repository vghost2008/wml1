#coding=utf-8
import tensorflow as tf
import wmodule
from .build import META_ARCH_REGISTRY
from object_detection2.modeling.build import build_outputs
from object_detection2.modeling.backbone.build import build_backbone
from object_detection2.modeling.box_regression import FCOSBox2BoxTransform
import math
from object_detection2.standard_names import *
from .meta_arch import MetaArch
from object_detection2.datadef import *
import object_detection2.od_toolkit as odtk
from object_detection2.modeling.onestage_heads.fcos_head import FCOSHead

slim = tf.contrib.slim

__all__ = ["FCOS"]


@META_ARCH_REGISTRY.register()
class FCOS(MetaArch):
    """
    Implement FCOS: Fully Convolutional One-Stage Object Detection
    """

    def __init__(self, cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)

        # fmt: off
        self.num_classes              = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features              = cfg.MODEL.FCOS.IN_FEATURES
        # fmt: on

        self.backbone = build_backbone(cfg,parent=self,*args,**kwargs)

        self.head = FCOSHead(cfg=cfg.MODEL.FCOS,
                             parent=self,
                             *args,**kwargs)

        # Matching and loss
        self.box2box_transform = FCOSBox2BoxTransform()

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
        pred_logits, pred_regression,pred_center_ness = self.head(features)
        gt_boxes = batched_inputs[GT_BOXES]
        gt_length = batched_inputs[GT_LENGTH]
        gt_labels = batched_inputs[GT_LABELS]

        outputs = build_outputs(name=self.cfg.MODEL.FCOS.OUTPUTS,
            cfg=self.cfg.MODEL.FCOS,
            parent=self,
            box2box_transform=self.box2box_transform,
            pred_logits=pred_logits,
            pred_regression =pred_regression,
            pred_center_ness=pred_center_ness,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_length=gt_length,
            batched_inputs=batched_inputs,
            max_detections_per_image=self.cfg.TEST.DETECTIONS_PER_IMAGE,
        )

        if self.is_training:
            if self.cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
                results = outputs.inference(inputs=batched_inputs,box_cls=pred_logits,
                                            box_regression=pred_regression,
                                            center_ness=pred_center_ness)
            else:
                results = {}

            return results,outputs.losses()
        else:
            results = outputs.inference(inputs=batched_inputs,box_cls=pred_logits,
                                        box_regression=pred_regression,
                                        center_ness=pred_center_ness)
            return results,{}



