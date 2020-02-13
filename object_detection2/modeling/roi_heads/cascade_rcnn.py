#coding=utf-8
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..poolers import ROIPooler
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from .roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from object_detection2.datadef import *
import wml_tfutils as wmlt
import wnnlayer as wnnl


@ROI_HEADS_REGISTRY.register()
class CascadeROIHeads(StandardROIHeads):
    def _init_box_head(self, cfg,*args,**kwargs):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        bin_size = cfg.MODEL.ROI_BOX_HEAD.bin_size
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        cascade_ious             = cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS
        self.num_cascade_stages  = len(cascade_ious)
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES

        assert len(cascade_bbox_reg_weights) == self.num_cascade_stages
        assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,  \
            "CascadeROIHeads only support class-agnostic regression now!"
        assert cascade_ious[0] == cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0]
        # fmt: on

        self.box_pooler = ROIPooler(cfg=cfg.MODEL.ROI_BOX_HEAD,parent=self,
            output_size=pooler_resolution,
            pooler_type=pooler_type,
            bin_size=bin_size,
            *args,**kwargs
        )

        self.box_head = []
        self.box_predictor = []
        self.box2box_transform = []
        self.proposal_matchers = []
        for k in range(self.num_cascade_stages):
            box_head = build_box_head(cfg, parent=self,*args,**kwargs)
            self.box_head.append(box_head)
            self.box_predictor.append(
                FastRCNNOutputLayers(cfg,parent=self,num_classes=self.num_classes,
                    cls_agnostic_bbox_reg=True,**kwargs
                )
            )
            self.box2box_transform.append(Box2BoxTransform(weights=cascade_bbox_reg_weights[k]))

            if k == 0:
                # The first matching is done by the matcher of ROIHeads (self.proposal_matcher).
                self.proposal_matchers.append(None)
            else:
                self.proposal_matchers.append(
                    Matcher([cascade_ious[k]], allow_low_quality_matches=False,cfg=cfg,parent=self)
                )

    def forward(self, inputs, features, proposals: ProposalsData):
        proposals_boxes = proposals.boxes
        if self.is_training:
            proposals = self.label_and_sample_proposals(inputs,proposals_boxes)

        features_list = [features[f] for f in self.in_features]

        if self.is_training:
            # Need targets to box head
            losses = self._forward_box(inputs,features_list, proposals)
            if self.train_on_pred_boxes:
                # proposals里面的box已经是采样的结果,无需再进行采样操作
                proposals = self.label_and_sample_proposals(inputs, proposals.boxes, do_sample=False)
            losses.update(self._forward_mask(inputs,features_list, proposals))
            losses.update(self._forward_keypoint(inputs,features_list, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(inputs,features_list, proposals)
            pred_instances = self.forward_with_given_boxes(inputs,features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, inputs,features, proposals):
        head_outputs = []
        for k in range(self.num_cascade_stages):
            if k > 0:
                # The output boxes of the previous stage are the input proposals of the next stage
                if self.is_training:
                    proposals = self.label_and_sample_proposals(inputs,head_outputs[-1].predict_boxes_for_gt_classes(),do_sample=False)
            head_outputs.append(self._run_stage(features, proposals, k))

        if self.is_training:
            losses = {}
            for stage, output in enumerate(head_outputs):
                stage_losses = output.losses()
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h.predict_probs() for h in head_outputs]

            # Average the scores across heads
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            # Use the boxes of the last head
            return head_outputs[-1].inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )

    def _run_stage(self, features, proposals, stage):
        """
        Args:
            features (list[Tensor]): #lvl input features to ROIHeads
            proposals (list[Instances]): #image Instances, with the field "proposal_boxes"
            stage (int): the current stage

        Returns:
            FastRCNNOutputs: the output of this stage
        """
        box_features = self.box_pooler(features, proposals.boxes)
        # The original implementation averages the losses among heads,
        # but scale up the parameter gradients of the heads.
        # This is equivalent to adding the losses among heads,
        # but scale down the gradients on features.
        box_features = wnnl.scale_gradient(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_head[stage](box_features,scope=f"BoxHead{stage}")
        pred_class_logits, pred_proposal_deltas = self.box_predictor[stage](box_features,scope=f"BoxPredictor{stage}")
        del box_features

        outputs = FastRCNNOutputs(
            cfg=self.cfg,
            parent=self,
            box2box_transform=self.box2box_transform[stage],
            pred_class_logits=pred_class_logits,
            pred_proposal_deltas=pred_proposal_deltas,
            proposals=proposals,
        )
        return outputs
