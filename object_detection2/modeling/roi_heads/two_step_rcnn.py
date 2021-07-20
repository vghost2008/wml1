#coding=utf-8
from .roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
import basic_tftools as btf
from object_detection2.datadef import *
from .build import *
from object_detection2.odtools import *
from .box_head import BoxesForwardType

@ROI_HEADS_REGISTRY.register()
class TwoStepROIHeads(StandardROIHeads):
    def _forward_box(self, features, proposals,img_size,retry=True):
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
        assert self.cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG, "Only support cls agnostic bbox reg."

        if self.is_training:
            '''
            During training, the quality of rcnn boxes is not guaranted.
            '''
            return StandardROIHeads._forward_box(features,proposals,img_size,retry)
        proposal_boxes = proposals[PD_BOXES] #when inference proposals's a dict which is the outputs of RPN

        self.t_proposal_boxes = proposal_boxes
        self.t_img_size = img_size

        box_features = self.box_pooler(features, proposal_boxes,img_size=img_size)
        if self.roi_hook is not None:
            box_features = self.roi_hook(box_features,self.batched_inputs,fwd_type=BoxesForwardType.BBOXES)
        box_features = self.box_head(box_features,fwd_type=BoxesForwardType.BBOXES)
        if self.cfg.MODEL.ROI_HEADS.PRED_IOU:
            _, pred_proposal_deltas,_ = self.box_predictor(box_features,fwd_type=BoxesForwardType.BBOXES)
        else:
            _, pred_proposal_deltas = self.box_predictor(box_features,fwd_type=BoxesForwardType.BBOXES)

        shape = btf.combined_static_and_dynamic_shape(proposal_boxes)
        proposal_boxes = self.box2box_transform.apply_deltas(deltas=tf.reshape(pred_proposal_deltas,shape),
                                                             boxes=proposal_boxes)


        proposal_boxes = tf.stop_gradient(proposal_boxes)
        self.t_proposal_boxes = proposal_boxes
        box_features = self.box_pooler(features, proposal_boxes,img_size=img_size)
        if self.roi_hook is not None:
            box_features = self.roi_hook(box_features,self.batched_inputs,fwd_type=BoxesForwardType.IOUS|BoxesForwardType.CLASSES)
        box_features = self.box_head(box_features,fwd_type=BoxesForwardType.IOUS|BoxesForwardType.CLASSES)
        if self.cfg.MODEL.ROI_HEADS.PRED_IOU:
            pred_class_logits, _,iou_logits = self.box_predictor(box_features,
                                                                 fwd_type=BoxesForwardType.IOUS|BoxesForwardType.CLASSES)
        else:
            pred_class_logits, _ = self.box_predictor(box_features,
                                                      fwd_type = BoxesForwardType.IOUS | BoxesForwardType.CLASSES)
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
        pred_instances = outputs.inference(
            self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img,
            pred_iou_logits = iou_logits,
        )
        '''if self.cfg.MODEL.ROI_HEADS.PRED_IOU and retry:
            proposals[PD_BOXES] = pred_instances[RD_BOXES]
            scope = tf.get_variable_scope()
            scope.reuse_variables()
            return self._forward_box(features,proposals,img_size,retry=False)'''

        return pred_instances,{}


