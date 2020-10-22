#coding=utf-8
import tensorflow as tf
from .build import META_ARCH_REGISTRY
from object_detection2.modeling.build import build_outputs
from object_detection2.modeling.backbone.build import build_backbone
from object_detection2.modeling.anchor_generator import build_anchor_generator
from object_detection2.modeling.box_regression import Box2BoxTransform,OffsetBox2BoxTransform
from object_detection2.modeling.build_matcher import build_matcher
from object_detection2.standard_names import *
from object_detection2.modeling.onestage_heads.retinanet_outputs import *
from .meta_arch import MetaArch
from object_detection2.datadef import *
from object_detection2.modeling.onestage_heads.build import build_retinanet_head

slim = tf.contrib.slim

__all__ = ["RetinaNet"]

@META_ARCH_REGISTRY.register()
class RetinaNet(MetaArch):
    """
    Implement RetinaNet (https://arxiv.org/abs/1708.02002).
    """

    def __init__(self, cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)

        # fmt: off
        self.num_classes              = cfg.MODEL.RETINANET.NUM_CLASSES
        self.in_features              = cfg.MODEL.RETINANET.IN_FEATURES
        # fmt: on

        self.backbone = build_backbone(cfg,parent=self,*args,**kwargs)

        self.anchor_generator = build_anchor_generator(cfg,parent=self,*args,**kwargs)
        self.head = build_retinanet_head(cfg.MODEL.RETINANET.HEAD_NAME,cfg=cfg.MODEL.RETINANET,
                                  num_anchors=self.anchor_generator.num_cell_anchors,
                                  parent=self,
                                  *args,**kwargs)

        # Matching and loss
        #self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.box2box_transform = OffsetBox2BoxTransform()
        self.anchor_matcher = build_matcher(
            cfg.MODEL.RETINANET.MATCHER,
            thresholds=cfg.MODEL.RETINANET.IOU_THRESHOLDS,
            allow_low_quality_matches=True,
            cfg=cfg,
            parent=self,
            k = self.anchor_generator.num_cell_anchors[0],
        )


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
        pred_logits, pred_anchor_deltas= self.head(features)
        anchors = self.anchor_generator(batched_inputs,features)
        gt_boxes = batched_inputs.get(GT_BOXES,None)
        gt_length = batched_inputs.get(GT_LENGTH,None)
        gt_labels = batched_inputs.get(GT_LABELS,None)

        outputs = build_outputs(name=self.cfg.MODEL.RETINANET.OUTPUTS,
            cfg=self.cfg.MODEL.RETINANET,
            parent=self,
            box2box_transform=self.box2box_transform,
            anchor_matcher=self.anchor_matcher,
            pred_logits=pred_logits,
            pred_anchor_deltas=pred_anchor_deltas,
            anchors=anchors,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_length=gt_length,
            max_detections_per_image=self.cfg.TEST.DETECTIONS_PER_IMAGE,
        )
        outputs.batched_inputs = batched_inputs

        if self.is_training:
            if self.cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
                results = outputs.inference(inputs=batched_inputs,box_cls=pred_logits,
                                            box_delta=pred_anchor_deltas, anchors=anchors)
            else:
                results = {}

            return results,outputs.losses()
        else:
            results = outputs.inference(inputs=batched_inputs,box_cls=pred_logits,
                                        box_delta=pred_anchor_deltas, anchors=anchors)
            return results,{}
