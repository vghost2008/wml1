#coding=utf-8
from .roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
import basic_tftools as btf
from object_detection2.datadef import *
from .build import *
from object_detection2.odtools import *
import wml_tfutils as wmlt
import wsummary
import object_detection2.od_toolkit as odt
slim = tf.contrib.slim


@ROI_HEADS_REGISTRY.register()
class GEROIHeads(StandardROIHeads):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.rcnn_anchor_boxes = tf.reshape(tf.convert_to_tensor(self.cfg.MODEL.ANCHOR_GENERATOR.SIZES,dtype=tf.float32),[1,-1])
        self.normalizer_fn, self.norm_params = odt.get_norm(self.cfg.MODEL.ROI_BOX_HEAD.NORM, self.is_training)
        self.activation_fn = odt.get_activation_fn(self.cfg.MODEL.ROI_BOX_HEAD.ACTIVATION_FN)
        self.norm_scope_name = odt.get_norm_scope_name(self.cfg.MODEL.ROI_BOX_HEAD.NORM)

    @wmlt.add_name_scope
    def trans_boxes(self,bboxes,levels,img_size):
        B,box_nr = wmlt.combined_static_and_dynamic_shape(levels)
        anchor_boxes_size = tf.tile(self.rcnn_anchor_boxes,[B,1])
        boxes_size = wmlt.batch_gather(anchor_boxes_size,levels)
        w = boxes_size/tf.to_float(img_size[1])
        h = boxes_size/tf.to_float(img_size[0])
        ymin,xmin,ymax,xmax = tf.unstack(bboxes,axis=-1)
        cy = (ymin+ymax)/2
        cx = (xmin+xmax)/2
        ymin = cy-h/2
        ymax= cy+h/2
        xmin = cx-w/2
        xmax= cx+w/2
        new_boxes = tf.stack([ymin,xmin,ymax,xmax],axis=-1)
        #####
        log_bboxes = tf.concat([bboxes[:,:3],new_boxes[:,:3]],axis=1)
        log_labels = tf.convert_to_tensor([[1,2,3,11,12,13]],dtype=tf.int32)
        log_labels = tf.tile(log_labels,[B,1])
        wsummary.detection_image_summary(self.batched_inputs[IMAGE],boxes=log_bboxes,classes=log_labels,
                                         name="to_anchor_bboxes")
        return new_boxes


    def _forward_box(self, features, proposals, img_size, retry=True):
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
        if self.is_training:
            proposal_boxes = proposals.boxes  # when training proposals's EncodedData
        else:
            proposal_boxes = proposals[PD_BOXES]  # when inference proposals's a dict which is the outputs of RPN
        self.t_proposal_boxes = proposal_boxes
        features = self.forward_features(features)
        cls_x = self.box_pooler(features[0], proposal_boxes, img_size=img_size)
        box_x = self.box_pooler(features[1], proposal_boxes, img_size=img_size)

        new_bboxes = self.trans_boxes(proposal_boxes, self.box_pooler.level_assignments,
                                           img_size)  # when training proposals's EncodedData
        del proposal_boxes
        if self.is_training:
            proposals.boxes = new_bboxes  # when training proposals's EncodedData
        else:
            proposals[PD_BOXES]  = new_bboxes # when inference proposals's a dict which is the outputs of RPN

        box_features = [cls_x,box_x]
        if self.cfg.MODEL.ROI_HEADS.PRED_IOU:
            pred_class_logits, pred_proposal_deltas, iou_logits = self.box_predictor(box_features)
        else:
            pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
            iou_logits = None
        del box_features

        outputs = build_outputs(name=self.cfg.MODEL.ROI_HEADS.OUTPUTS,
                                cfg=self.cfg, parent=self,
                                box2box_transform=self.box2box_transform,
                                pred_class_logits=pred_class_logits,
                                pred_proposal_deltas=pred_proposal_deltas,
                                pred_iou_logits=iou_logits,
                                proposals=proposals,
                                )
        if self.is_training:
            if self.train_on_pred_boxes:
                pred_boxes = outputs.predict_boxes_for_gt_classes()
                self.rcnn_outboxes = pred_boxes
            if self.cfg.GLOBAL.SUMMARY_LEVEL <= SummaryLevel.DEBUG:
                pred_instances = outputs.inference(
                    self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img,
                    pred_iou_logits=iou_logits,
                    proposal_boxes=proposals.boxes
                )
            else:
                pred_instances = {}
            return pred_instances, outputs.losses()
        else:
            pred_instances = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img,
                pred_iou_logits=iou_logits,
            )
            '''if self.cfg.MODEL.ROI_HEADS.PRED_IOU and retry:
                proposals[PD_BOXES] = pred_instances[RD_BOXES]
                scope = tf.get_variable_scope()
                scope.reuse_variables()
                return self._forward_box(features,proposals,img_size,retry=False)'''

            return pred_instances, {}

    def forward_features(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, Hi, Wi,AxK).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Hi, Wi, Ax4).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        num_convs = 4
        logits = []
        bbox_reg = []
        self.logits_pre_outputs = []
        self.bbox_reg_pre_outputs = []
        for j, feature in enumerate(features):
            channels = feature.get_shape().as_list()[-1]
            with tf.variable_scope("WeightSharedConvolutionalBoxPredictor", reuse=tf.AUTO_REUSE):
                net = feature
                with tf.variable_scope("BoxPredictionTower"):
                    for i in range(num_convs):
                        net = slim.conv2d(net, channels, [3, 3],
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          biases_initializer=None if self.normalizer_fn is not None else tf.zeros_initializer(),
                                          scope=f"conv2d_{i}")
                        if self.normalizer_fn is not None:
                            with tf.variable_scope(f"conv2d_{i}"):
                                net = self.normalizer_fn(net, scope=f'{self.norm_scope_name}/feature_{j}',
                                                         **self.norm_params)
                        if self.activation_fn is not None:
                            net = self.activation_fn(net)
                _bbox_reg = net

                net = feature
                with tf.variable_scope("ClassPredictionTower"):
                    for i in range(num_convs):
                        net = slim.conv2d(net, channels, [3, 3],
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          biases_initializer=None if self.normalizer_fn is not None else tf.zeros_initializer(),
                                          scope=f"conv2d_{i}")
                        if self.normalizer_fn is not None:
                            with tf.variable_scope(f"conv2d_{i}"):
                                net = self.normalizer_fn(net, scope=f'{self.norm_scope_name}/feature_{j}',
                                                         **self.norm_params)
                        if self.activation_fn is not None:
                            net = self.activation_fn(net)
                _logits = net

            logits.append(_logits)
            bbox_reg.append(_bbox_reg)
        return logits, bbox_reg


