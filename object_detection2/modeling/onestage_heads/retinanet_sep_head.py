#coding=utf-8
import tensorflow as tf
import wmodule
from .build import RETINANET_HEAD
from object_detection2.modeling.build import build_outputs
from object_detection2.modeling.backbone.build import build_backbone
from object_detection2.modeling.anchor_generator import build_anchor_generator
from object_detection2.modeling.box_regression import Box2BoxTransform
from object_detection2.modeling.build_matcher import build_matcher
from .onestage_tools import general_to_N_HWA_K_and_concat
import math
from object_detection2.standard_names import *
from object_detection2.modeling.onestage_heads.retinanet_outputs import *
from object_detection2.datadef import *
import object_detection2.od_toolkit as odtk
import wnnlayer as wnnl

slim = tf.contrib.slim
@RETINANET_HEAD.register()
class RetinaNetSepHead(wmodule.WChildModule):
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
        # Detectron2默认没有使用normalizer, 但在测试数据集上发现不使用normalizer网络不收敛
        self.normalizer_fn, self.norm_params = odtk.get_norm(self.cfg.NORM, is_training=self.is_training)
        self.activation_fn = odtk.get_activation_fn(self.cfg.ACTIVATION_FN)
        self.norm_scope_name = odtk.get_norm_scope_name(self.cfg.NORM)
        self.logits_before_outputs = []
        self.regs_before_outputs = []

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
        num_convs = cfg.NUM_CONVS
        prior_prob = cfg.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        logits = []
        bbox_reg = []
        logits_before_outputs_list = []
        regs_before_outputs_list = []
        for j, feature in enumerate(features):
            channels = feature.get_shape().as_list()[-1]
            last_channel = 96
            with tf.variable_scope("WeightSharedConvolutionalBoxPredictor", reuse=tf.AUTO_REUSE):
                net = feature
                with tf.variable_scope("BoxPredictionTower"):
                    for i in range(num_convs):
                        if i==num_convs-1:
                            cur_channels = self.num_anchors*last_channel
                        else:
                            cur_channels = channels
                        net = slim.conv2d(net, cur_channels, [3, 3],
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
                regs_before_outputs_list.append(net)
                B,H,W,C = wmlt.combined_static_and_dynamic_shape(net)
                net = tf.reshape(net,[B,H,W,self.num_anchors,last_channel])
                net = tf.transpose(net,[0,3,1,2,4])
                net = tf.reshape(net,[B*self.num_anchors,H,W,last_channel])
                net = slim.conv2d(net, 4, [3, 3], activation_fn=None,
                                        normalizer_fn=None,
                                        scope="BoxPredictor")
                net = tf.reshape(net,[B,self.num_anchors,H,W,4])
                net = tf.transpose(net,[0,2,3,1,4])
                _bbox_reg = tf.reshape(net,[B,H,W,self.num_anchors*4])

                net = feature
                with tf.variable_scope("ClassPredictionTower"):
                    for i in range(num_convs):
                        if i==num_convs-1:
                            cur_channels = self.num_anchors*last_channel
                        else:
                            cur_channels = channels
                        net = slim.conv2d(net, cur_channels, [3, 3],
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
                logits_before_outputs_list.append(net)
                B,H,W,C = wmlt.combined_static_and_dynamic_shape(net)
                net = tf.reshape(net,[B,H,W,self.num_anchors,last_channel])
                net = tf.transpose(net,[0,3,1,2,4])
                net = tf.reshape(net,[B*self.num_anchors,H,W,last_channel])
                net = slim.conv2d(net, num_classes, [3, 3], activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=tf.constant_initializer(value=bias_value),
                                  scope="ClassesPredictor")
                net = tf.reshape(net,[B,self.num_anchors,H,W,num_classes])
                net = tf.transpose(net,[0,2,3,1,4])
                _logits = tf.reshape(net,[B,H,W,self.num_anchors*num_classes])

            logits.append(_logits)
            bbox_reg.append(_bbox_reg)

        self.logits_before_outputs = general_to_N_HWA_K_and_concat(logits_before_outputs_list,K=last_channel)
        self.regs_before_outputs = general_to_N_HWA_K_and_concat(regs_before_outputs_list,K=last_channel)
        return logits, bbox_reg