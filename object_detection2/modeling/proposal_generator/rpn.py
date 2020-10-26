#coding=utf-8
import tensorflow as tf
from thirdparty.registry import Registry
import wmodule
from .build import PROPOSAL_GENERATOR_REGISTRY
from object_detection2.modeling.anchor_generator import *
from object_detection2.modeling.box_regression import *
from .rpn_outputs import find_top_rpn_proposals
from object_detection2.datadef import *
import wsummary
from object_detection2.modeling.build import build_outputs
from object_detection2.modeling.build_matcher import build_matcher
import object_detection2.od_toolkit as odtk
from object_detection2.modeling.backbone.build import build_hook_by_name

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
        num_cell_anchors = self.anchor_generator.num_cell_anchors
        assert len(set(num_cell_anchors))==1,"all levers cell anchors num must be equal."
        self.num_cell_anchors = num_cell_anchors[0]
        self.box_dim = self.anchor_generator.box_dim
        self.normalizer_fn,self.norm_params = odtk.get_norm(self.cfg.MODEL.RPN.NORM,is_training=self.is_training)
        self.activation_fn = odtk.get_activation_fn(self.cfg.MODEL.RPN.ACTIVATION_FN)
        self.hook = build_hook_by_name(self.cfg.MODEL.RPN.HOOK,self.cfg,parent=self)

    def forward(self,inputs,features):
        '''

        :param inputs:
        :param features: list of [N,Hi,Wi,Ci] Ci must be equal.
        :return:
        logits: list of [N,Hi,Wi,A]
        anchor_deltas: list of [N,Hi,Wi,Ax4]
        '''
        del inputs
        pred_objectness_logits = []
        pred_anchor_deltas = []

        if self.hook is not None:
            features = self.hook(features,{})

        for i,x in enumerate(features):
            channel = x.get_shape()[-1]
            with tf.variable_scope("StandardRPNHead",reuse=tf.AUTO_REUSE):
                t = slim.conv2d(x,channel,[3,3],normalizer_fn=None,
                                activation_fn=None,
                                padding="SAME")

                if self.normalizer_fn is not None:
                    with tf.variable_scope(f"norm{i}"):
                        t = self.normalizer_fn(t,**self.norm_params)

                if self.activation_fn is not None:
                    t = self.activation_fn(t)

                t0 = slim.conv2d(t,self.num_cell_anchors,[1,1],activation_fn=None,
                                 normalizer_fn=None,scope="objectness_logits")
                t1 = slim.conv2d(t,self.num_cell_anchors*self.box_dim,[1,1],activation_fn=None,
                                 normalizer_fn=None,scope="anchor_deltas")
                pred_objectness_logits.append(t0)
                pred_anchor_deltas.append(t1)

        return pred_objectness_logits,pred_anchor_deltas


@PROPOSAL_GENERATOR_REGISTRY.register()
class RPN(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)
        self.rpn_head = build_rpn_head(cfg,parent=self,*args,**kwargs)
        self.anchor_matcher = build_matcher(
            cfg.MODEL.RPN.MATCHER,
            thresholds=cfg.MODEL.RPN.IOU_THRESHOLDS,
            same_pos_label=1,
            allow_low_quality_matches=True,
            cfg=cfg,
            parent=self,
            k = 9*self.rpn_head.anchor_generator.num_cell_anchors[0],
        )
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
        self.anchors_num_per_level = []

    def forward(self,inputs,features):

        features = [features[f] for f in self.in_features]

        gt_boxes = inputs.get(GT_BOXES,None)
        #gt_labels = inputs.gt_labels
        gt_length = inputs.get(GT_LENGTH,None)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(inputs,features)
        anchors = self.rpn_head.anchor_generator(inputs,features)
        self.anchors_num_per_level = [wmlt.combined_static_and_dynamic_shape(x)[0] for x in anchors]
        outputs = build_outputs(self.cfg.MODEL.RPN.OUTPUTS,
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
        if self.cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
            outputs.inputs = inputs

        if self.is_training:
            losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}
            rpn_threshold = 0.0
        else:
            rpn_threshold = self.cfg.MODEL.PROPOSAL_GENERATOR.SCORE_THRESH_TEST
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
            self.anchors_num_per_level,
            score_threshold=rpn_threshold,
            is_training=self.is_training
        )
        if self.cfg.MODEL.RPN.SORT_RESULTS:
            with tf.name_scope("sort_rpn_results"):
                def fn(bboxes,keys):
                    N = wmlt.combined_static_and_dynamic_shape(keys)
                    new_keys,indices = tf.nn.top_k(keys,k=N[0])
                    bboxes = tf.gather(bboxes,indices)
                    return [bboxes,keys]
                proposals,logits = tf.map_fn(lambda x:fn(x[0],x[1]),elems=[proposals,logits],back_prop=False)

        outdata = {PD_BOXES:proposals,PD_PROBABILITY:tf.nn.sigmoid(logits)}
        wsummary.detection_image_summary(images=inputs[IMAGE],boxes=outdata[PD_BOXES],
                                         name="rpn/proposals")

        return outdata,losses

