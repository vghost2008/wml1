#coding=utf-8
import tensorflow as tf
import wmodule
from .build import RETINANET_HEAD
from object_detection2.modeling.build import build_outputs
from object_detection2.modeling.backbone.build import build_backbone
from object_detection2.modeling.anchor_generator import build_anchor_generator
from object_detection2.modeling.box_regression import Box2BoxTransform
from object_detection2.modeling.build_matcher import build_matcher
import math
from object_detection2.standard_names import *
from object_detection2.modeling.onestage_heads.retinanet_outputs import *
from object_detection2.datadef import *
import object_detection2.od_toolkit as odtk

slim = tf.contrib.slim
@RETINANET_HEAD.register()
class YOLOHead(wmodule.WChildModule):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, num_anchors, cfg, parent, *args, **kwargs):
        '''

        :param num_anchors:
        :param cfg:  only the child part
        :param parent:
        :param args:
        :param kwargs:
        '''
        super().__init__(cfg, *args, parent=parent, **kwargs)
        assert (
                len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        self.num_anchors = num_anchors[0]
        self.normalizer_fn, self.norm_params = odtk.get_norm(self.cfg.NORM, is_training=self.is_training)
        self.activation_fn = odtk.get_activation_fn(self.cfg.ACTIVATION_FN)
        self.norm_scope_name = odtk.get_norm_scope_name(self.cfg.NORM)
        self.logits_pre_outputs = []
        self.bbox_reg_pre_outputs = []

    def conv_block(self,x,chs,three_conv=False,name=None):
        with tf.variable_scope(name,default_name="conv_block"):
            x = slim.conv2d(x, chs, 1,
                            normalizer_fn=self.normalizer_fn,
                            normalizer_params=self.norm_params,
                            activation_fn=self.activation_fn
                            )

            x = slim.conv2d(x, chs*2, 3,
                            normalizer_fn=self.normalizer_fn,
                            normalizer_params=self.norm_params,
                            activation_fn=self.activation_fn
                            )
            if three_conv:
                x = slim.conv2d(x, chs, 1,
                                normalizer_fn=self.normalizer_fn,
                                normalizer_params=self.norm_params,
                                activation_fn=self.activation_fn
                                )
            return x

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
        num_classes = cfg.NUM_CLASSES
        prior_prob = cfg.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        logits = []
        bbox_reg = []
        self.logits_pre_outputs = []
        self.bbox_reg_pre_outputs = []
        assert len(features)==3,"Error features len."
        channels = [128,256,512]
        with tf.variable_scope("ConvolutionalBoxPredictor", reuse=tf.AUTO_REUSE):
            for i,chs in enumerate(channels):
                with tf.variable_scope(f"Head{i}"):
                    net3 = self.conv_block(features[i],chs=chs,name="block0")
                    net3 = self.conv_block(net3,chs=chs,name="block1")
                    net3 = self.conv_block(net3,chs=chs,name="block2")
                    _bbox_reg = slim.conv2d(net3, self.num_anchors * 4, [3, 3], activation_fn=None,
                                            normalizer_fn=None,
                                            scope="BoxPredictor")
                    _logits = slim.conv2d(net3, self.num_anchors * num_classes, [3, 3], activation_fn=None,
                                          normalizer_fn=None,
                                          biases_initializer=tf.constant_initializer(value=bias_value),
                                          scope="ClassPredictor")
                logits.append(_logits)
                bbox_reg.append(_bbox_reg)

        return logits, bbox_reg