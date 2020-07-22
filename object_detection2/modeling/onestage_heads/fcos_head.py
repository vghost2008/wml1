#coding=utf-8
import tensorflow as tf
import wmodule
import math
import object_detection2.od_toolkit as odtk
import wsummary
import wnnlayer as wnnl

slim = tf.contrib.slim

class FCOSHead(wmodule.WChildModule):
    """
    The head used in FCOS for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg,parent,*args,**kwargs):
        '''

        :param cfg:  only the child part
        :param parent:
        :param args:
        :param kwargs:
        '''
        super().__init__(cfg,*args,parent=parent,**kwargs)
        self.normalizer_fn,self.norm_params = odtk.get_norm(self.cfg.NORM,is_training=self.is_training)
        self.activation_fn = odtk.get_activation_fn(self.cfg.ACTIVATION_FN)
        self.norm_scope_name = odtk.get_norm_scope_name(self.cfg.NORM)
    @staticmethod
    def clip_exp(x):
        x = tf.minimum(x,9.3)
        return tf.exp(x)

    @staticmethod
    def clip_exp(x):
        x = tf.minimum(x,9.3)
        return tf.exp(x)

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
        center_ness = []
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
                _bbox_reg = slim.conv2d(net, 4, [3, 3], activation_fn=None,
                                         normalizer_fn=None,
                                         scope="BoxPredictor")
                _bbox_reg = _bbox_reg*wnnl.scale_gradient(tf.get_variable(name=f"gamma_{j}",shape=(),initializer=tf.ones_initializer()),0.2)
                #_bbox_reg = tf.nn.relu(_bbox_reg)
                _bbox_reg = self.clip_exp(_bbox_reg)
                _bbox_reg = _bbox_reg*math.pow(2,j)
                wsummary.variable_summaries_v2(_bbox_reg,"bbox_reg_net")

                '''net = feature
                with tf.variable_scope("CenterPredictionTower"):
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
                            net = self.activation_fn(net)'''
                _center_ness = slim.conv2d(net, 1, [3, 3], activation_fn=None,
                                           normalizer_fn=None,
                                           scope="CenterNessPredictor")
                _center_ness = tf.squeeze(_center_ness,axis=-1)

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
                _logits = slim.conv2d(net, num_classes, [3, 3], activation_fn=None,
                                         normalizer_fn=None,
                                         biases_initializer=tf.constant_initializer(value=bias_value),
                                         scope="ClassPredictor")

            logits.append(_logits)
            bbox_reg.append(_bbox_reg)
            center_ness.append(_center_ness)
        return logits, bbox_reg,center_ness
