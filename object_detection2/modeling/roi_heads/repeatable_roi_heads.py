#coding=utf-8
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..poolers import ROIPooler
from .build import build_box_head,build_outputs
from .fast_rcnn import FastRCNNOutputs
from .fast_rcnn_output_layers import FastRCNNOutputLayers
from .roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from object_detection2.datadef import *
import wml_tfutils as wmlt
import wnnlayer as wnnl
import wsummary
from object_detection2.odtools import *


@ROI_HEADS_REGISTRY.register()
class RepeatableROIHeads(StandardROIHeads):

    def forward(self, inputs, features, proposals: ProposalsData):
        """
        See :class:`ROIHeads.forward`.
        """
        self.batched_inputs = inputs
        proposals_boxes = proposals[PD_BOXES]
        if self.is_training:
            proposals = self.label_and_sample_proposals(inputs,proposals_boxes)

        features_list = [features[f] for f in self.in_features]

        img_size = get_img_size_from_batched_inputs(inputs)
        if self.is_training:
            pred_instances,losses = self._forward_box(features_list, proposals,img_size=img_size)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            if self.train_on_pred_boxes:
                #proposals里面的box已经是采样的结果,无需再进行采样操作
                proposals = self.label_and_sample_proposals(inputs,proposals.boxes,do_sample=False)
            losses.update(self._forward_mask(inputs,features_list, proposals,img_size=img_size))
            losses.update(self._forward_keypoint(inputs,features_list, proposals,img_size=img_size))
            return pred_instances, losses
        else:
            if self.cfg.MODEL.ROI_HEADS.REPEAT_FORWARD_BOX > 1:
                for i in range(self.cfg.MODEL.ROI_HEADS.REPEAT_FORWARD_BOX):
                    pred_instances,_ = self._forward_box(features_list, proposals,img_size=img_size,reuse=i>0)
                    proposals[PD_BOXES] = pred_instances[RD_BOXES]
            else:
                pred_instances,_ = self._forward_box(features_list, proposals,img_size=img_size)
                
            del proposals
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            mk_pred_instances = self.forward_with_given_boxes(inputs,features, pred_instances,img_size=img_size)
            pred_instances.update(mk_pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, inputs,features, instances,img_size):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.is_training
        features = [features[f] for f in self.in_features]

        instances = self._forward_mask(inputs,features, instances,img_size=img_size)
        instances = self._forward_keypoint(inputs,features, instances,img_size=img_size)
        return instances

    def _forward_box(self, features, proposals,img_size,scope="forward_box",reuse=False):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        with tf.variable_scope(scope,reuse=reuse):
            if self.is_training:
                proposal_boxes = proposals.boxes #when training proposals's EncodedData
            else:
                proposal_boxes = proposals[PD_BOXES] #when inference proposals's a dict which is the outputs of RPN
            self.t_proposal_boxes = proposal_boxes
            box_features = self.box_pooler(features, proposal_boxes,img_size=img_size)
            if self.roi_hook is not None:
                box_features = self.roi_hook(box_features,self.batched_inputs)
            box_features = self.box_head(box_features)
            if self.cfg.MODEL.ROI_HEADS.PRED_IOU:
                pred_class_logits, pred_proposal_deltas,iou_logits = self.box_predictor(box_features)
            else:
                pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
                iou_logits = None
            del box_features
    
            outputs = build_outputs(name=self.cfg.MODEL.ROI_HEADS.OUTPUTS,
                cfg=self.cfg,parent=self,
                box2box_transform=self.box2box_transform,
                pred_class_logits=pred_class_logits,
                pred_proposal_deltas=pred_proposal_deltas,
                pred_iou_logits = iou_logits,
                proposals=proposals,
                )
            if self.is_training:
                if self.train_on_pred_boxes:
                    pred_boxes = outputs.predict_boxes_for_gt_classes()
                    self.rcnn_outboxes = pred_boxes
                if self.cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
                    pred_instances = outputs.inference(
                        self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img,
                        pred_iou_logits=iou_logits,
                        proposal_boxes=proposals.boxes
                    )
                else:
                    pred_instances = {}
                return pred_instances,outputs.losses()
            else:
                pred_instances = outputs.inference(
                    self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img,
                    pred_iou_logits = iou_logits,
                )
    
                return pred_instances,{}

