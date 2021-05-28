#coding=utf-8
import tensorflow as tf
from .build import META_ARCH_REGISTRY
from object_detection2.modeling.backbone.build import build_backbone
from object_detection2.modeling.mot_heads.build import *
from .meta_arch import MetaArch
from object_detection2.datadef import *
from object_detection2.modeling.build import build_outputs

slim = tf.contrib.slim

__all__ = ["MOT"]

@META_ARCH_REGISTRY.register()
class MOT(MetaArch):

    def __init__(self, cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)

        # fmt: off
        self.in_features              = cfg.MODEL.MOT.IN_FEATURES
        # fmt: on

        self.backbone = build_backbone(cfg,parent=self,*args,**kwargs)

        self.head = build_MOT_head(cfg.MODEL.MOT.HEAD_NAME,
                                         num_MOT=cfg.MODEL.MOT.NUM_MOT,
                                         cfg=cfg.MODEL.MOT,
                                         parent=self,
                                         *args,**kwargs)

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
        pred_maps = self.head(features)
        gt_boxes = batched_inputs.get(GT_BOXES,None)
        gt_length = batched_inputs.get(GT_LENGTH,None)
        gt_labels = batched_inputs.get(GT_LABELS,None)

        outputs = build_outputs(name=self.cfg.MODEL.MOT.OUTPUTS,
            cfg=self.cfg.MODEL.MOT,
            parent=self,
            pred_maps =pred_maps,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_length=gt_length,
            max_detections_per_image=self.cfg.TEST.DETECTIONS_PER_IMAGE,
        )
        outputs.batched_inputs = batched_inputs

        if self.is_training:
            if self.cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
                results = outputs.inference(inputs=batched_inputs,pred_maps=pred_maps)
            else:
                results = {}

            return results,outputs.losses()
        else:
            results = outputs.inference(inputs=batched_inputs,pred_maps=pred_maps)
            return results,{}
