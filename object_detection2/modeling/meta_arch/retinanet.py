#coding=utf-8
import tensorflow as tf
import wmodule
from .build import META_ARCH_REGISTRY
from object_detection2.modeling.build import build_outputs
from object_detection2.modeling.backbone.build import build_backbone
from object_detection2.modeling.anchor_generator import build_anchor_generator
from object_detection2.modeling.box_regression import Box2BoxTransform
from object_detection2.modeling.build_matcher import build_matcher
import math
from object_detection2.standard_names import *
from object_detection2.modeling.onestage_heads.retinanet_outputs import *
from .meta_arch import MetaArch
from object_detection2.datadef import *
import object_detection2.od_toolkit as odtk
import wnnlayer as wnnl

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
        self.head = RetinaNetHead(cfg=cfg.MODEL.RETINANET,
                                  num_anchors=self.anchor_generator.num_cell_anchors,
                                  parent=self,
                                  *args,**kwargs)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
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


class RetinaNetHead(wmodule.WChildModule):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, num_anchors,cfg,parent,*args,**kwargs):
        '''

        :param num_anchors:
        :param cfg:  only the child part
        :param parent:
        :param args:
        :param kwargs:
        '''
        super().__init__(cfg,*args,parent=parent,**kwargs)
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        self.num_anchors = num_anchors[0]
        # Detectron2默认没有使用normalizer, 但在测试数据集上发现不使用normalizer网络不收敛
        self.normalizer_fn,self.norm_params = odtk.get_norm(self.cfg.NORM,is_training=self.is_training)
        self.activation_fn = odtk.get_activation_fn(self.cfg.ACTIVATION_FN)
        self.norm_scope_name = odtk.get_norm_scope_name(self.cfg.NORM)

    def forward(self, features):
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
        cfg = self.cfg
        num_classes      = cfg.NUM_CLASSES
        num_convs        = cfg.NUM_CONVS
        prior_prob       = cfg.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        logits = []
        bbox_reg = []
        for j,feature in enumerate(features):
            channels = feature.get_shape().as_list()[-1]
            with tf.variable_scope("WeightSharedConvolutionalBoxPredictor", reuse=tf.AUTO_REUSE):
                net = feature
                with tf.variable_scope("BoxPredictionTower"):
                    for i in range(num_convs):
                        net = slim.conv2d(net,channels,[3,3],
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          biases_initializer=None if self.normalizer_fn is not None else tf.zeros_initializer(),
                                          scope=f"conv2d_{i}")
                        if self.normalizer_fn is not None:
                            with tf.variable_scope(f"conv2d_{i}"):
                                net = self.normalizer_fn(net, scope=f'{self.norm_scope_name}/feature_{j}',**self.norm_params)
                        if self.activation_fn is not None:
                            net = self.activation_fn(net)
                _bbox_reg = slim.conv2d(net, self.num_anchors* 4, [3, 3], activation_fn=None,
                                         normalizer_fn=None,
                                         scope="BoxPredictor")
                #_bbox_reg = _bbox_reg*wnnl.scale_gradient(tf.get_variable(name=f"gamma_{j}",shape=(),initializer=tf.ones_initializer()),0.2)
                
                net = feature
                with tf.variable_scope("ClassPredictionTower"):
                    for i in range(num_convs):
                        net = slim.conv2d(net,channels,[3,3],
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          biases_initializer=None if self.normalizer_fn is not None else tf.zeros_initializer(),
                                          scope=f"conv2d_{i}")
                        if self.normalizer_fn is not None:
                            with tf.variable_scope(f"conv2d_{i}"):
                                net = self.normalizer_fn(net, scope=f'{self.norm_scope_name}/feature_{j}',**self.norm_params)
                        if self.activation_fn is not None:
                            net = self.activation_fn(net)
                _logits = slim.conv2d(net, self.num_anchors* num_classes, [3, 3], activation_fn=None,
                                         normalizer_fn=None,
                                         biases_initializer=tf.constant_initializer(value=bias_value),
                                         scope="ClassPredictor")

            logits.append(_logits)
            bbox_reg.append(_bbox_reg)
        return logits, bbox_reg
