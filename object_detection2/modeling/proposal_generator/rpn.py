#coding=utf-8
import tensorflow as tf
from thirdparty.registry import Registry
import wmodule
from .build import PROPOSAL_GENERATOR_REGISTRY
from object_detection2.modeling.anchor_generator import *
from object_detection2.modeling.matcher import *
from object_detection2.modeling.box_regression import *
from .rpn_outputs import RPNOutputs,find_top_rpn_proposals
from object_detection2.datadef import *

slim = tf.contrib.slim

RPN_HEAD_REGISTRY = Registry("RPN_HEAD")

def build_rpn_head(cfg, *args,**kwargs):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, *args,**kwargs)

@RPN_HEAD_REGISTRY.register()
class StandardRPNHead(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)
        self.anchor_generator = build_anchor_generator(cfg,parent=self,*args,**kwargs)
        self.num_cell_anchors = self.anchor_generator.num_cell_anchors[0]
        self.box_dim = self.anchor_generator.box_dim

    def forward(self,inputs,features):
        del inputs
        pred_objectness_logits = []
        pred_anchor_deltas = []

        for x in features:
            channel = x.get_shape()[-1]
            t = slim.conv2d(x,channel,[3,3],normalizer_fn=None,
                          activation_fn=tf.nn.relu,
                          padding="SAME")
            t0 = slim.conv2d(t,self.num_cell_anchors,[1,1],activation_fn=None,
                             normalizer_fn=None)
            t1 = slim.conv2d(t,self.num_cell_anchors*self.box_dim,[1,1],activation_fn=None,
                             normalizer_fn=None)
            pred_objectness_logits.append(t0)
            pred_anchor_deltas.append(t1)

        return pred_objectness_logits,pred_anchor_deltas


@PROPOSAL_GENERATOR_REGISTRY.register()
class RPN(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)
        self.rpn_head = build_rpn_head(cfg,parent=self,*args,**kwargs)
        self.anchor_matcher = Matcher(thresholds=cfg.MODEL.RPN.IOU_THRESHOLDS,same_pos_label=1,allow_low_quality_matches=True,cfg=cfg,
                                      parent=self)
        self.box2box_transform = Box2BoxTransform()
        self.in_features             = cfg.MODEL.RPN.IN_FEATURES
        self.nms_thresh              = cfg.MODEL.RPN.NMS_THRESH
        self.batch_size_per_image    = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction       = cfg.MODEL.RPN.POSITIVE_FRACTION
        self.loss_weight             = cfg.MODEL.RPN.LOSS_WEIGHT
        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
        }

    def forward(self,inputs,features):

        features = [features[f] for f in self.in_features]

        gt_boxes = inputs['gt_boxes']
        #gt_labels = inputs.gt_labels
        gt_length = inputs['gt_length']

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(inputs,features)
        anchors = self.rpn_head.anchor_generator(inputs,features)

        outputs = RPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            gt_boxes,
            gt_length=gt_length
        )

        if self.is_training:
            losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}
        else:
            losses = {}

        # Find the top proposals by applying NMS and removing boxes that
        # are too small. The proposals are treated as fixed for approximate
        # joint training with roi heads. This approach ignores the derivative
        # w.r.t. the proposal boxes’ coordinates that are also network
        # responses, so is approximate.
        proposals,logits = find_top_rpn_proposals(
            outputs.predict_proposals(),
            outputs.predict_objectness_logits(),
            self.nms_thresh,
            self.pre_nms_topk[self.is_training],
            self.post_nms_topk[self.is_training],
        )

        outdata = {PD_BOXES:proposals,PD_PROBABILITY:tf.nn.sigmoid(logits)}

        return outdata,losses

