#coding=utf-8
from object_detection2.modeling.meta_arch.retinanet import RetinaNetHead as RetinaNetHead
from object_detection2.modeling.proposal_generator.retinanet_head import RetinaNetOutputs
from object_detection2.modeling.proposal_generator.rpn_outputs import find_top_rpn_proposals
from object_detection2.standard_names import *
import wml_tfutils as wmlt
import wsummary
from .build import PROPOSAL_GENERATOR_REGISTRY
import wmodule
from object_detection2.modeling.backbone.build import build_backbone
from object_detection2.modeling.anchor_generator import build_anchor_generator
from object_detection2.modeling.box_regression import Box2BoxTransform
from object_detection2.modeling.matcher import Matcher
import math
from object_detection2.datadef import *
'''
Use retinanet as a proposal generator
'''

@PROPOSAL_GENERATOR_REGISTRY.register()
class RetinaNet(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent=parent,*args,**kwargs)

        # fmt: off
        self.in_features = cfg.MODEL.RETINANET_PG.IN_FEATURES
        self.nms_thresh = cfg.MODEL.RETINANET_PG.NMS_THRESH
        self.loss_weight = cfg.MODEL.RETINANET_PG.LOSS_WEIGHT
        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.MODEL.RETINANET_PG.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RETINANET_PG.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.RETINANET_PG.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RETINANET_PG.POST_NMS_TOPK_TEST,
        }
        # fmt: on
        self.anchor_generator = build_anchor_generator(cfg,parent=self,*args,**kwargs)
        self.head = RetinaNetHead(cfg=cfg.MODEL.RETINANET_PG, num_anchors=self.anchor_generator.num_cell_anchors,parent=self,
                                  *args,**kwargs)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RETINANET_PG.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.RETINANET_PG.IOU_THRESHOLDS,
            allow_low_quality_matches=True,
            cfg=cfg,
            parent=self,
        )

    def forward(self, batched_inputs,features):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        features = [features[f] for f in self.in_features]
        pred_logits, pred_anchor_deltas= self.head(features)
        anchors = self.anchor_generator(batched_inputs,features)
        self.anchors_num_per_level = [wmlt.combined_static_and_dynamic_shape(x)[0] for x in anchors]
        gt_boxes = batched_inputs[GT_BOXES]
        gt_length = batched_inputs[GT_LENGTH]
        gt_labels = batched_inputs[GT_LABELS]

        outputs = RetinaNetOutputs(
            cfg=self.cfg.MODEL.RETINANET_PG,
            parent=self,
            box2box_transform=self.box2box_transform,
            anchor_matcher=self.anchor_matcher,
            pred_logits=pred_logits,
            pred_anchor_deltas=pred_anchor_deltas,
            anchors=anchors,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_length=gt_length,
            max_detections_per_image=self.cfg.TEST.DETECTIONS_PER_IMAGE
        )

        if self.is_training:
            if self.cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
                outputs.inference(inputs=batched_inputs,box_cls=pred_logits,
                                            box_delta=pred_anchor_deltas, anchors=anchors)
            proposals, logits = find_top_rpn_proposals(
                outputs.predict_proposals(),
                outputs.predict_objectness_logits(),
                self.nms_thresh,
                self.pre_nms_topk[self.is_training],
                self.post_nms_topk[self.is_training],
                self.anchors_num_per_level,
            )

            outdata = {PD_BOXES: proposals, PD_PROBABILITY: tf.nn.sigmoid(logits)}
            wsummary.detection_image_summary(images=batched_inputs[IMAGE], boxes=outdata[PD_BOXES],
                                             name="rpn/proposals")

            losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}
            return outdata, losses
        else:
            results = outputs.inference(inputs=batched_inputs,box_cls=pred_logits,
                                        box_delta=pred_anchor_deltas, anchors=anchors)

            assert batched_inputs[IMAGE].get_shape().as_list()[0]==1,"inference only support batch size equal one."

            length = results[RD_LENGTH][0]

            outdata = {PD_BOXES: results[RD_BOXES][:,:length], PD_PROBABILITY: results[RD_PROBABILITY][:,:length]}

            wsummary.detection_image_summary(images=batched_inputs[IMAGE], boxes=outdata[PD_BOXES],
                                             name="rpn/proposals")
            return outdata,{}
