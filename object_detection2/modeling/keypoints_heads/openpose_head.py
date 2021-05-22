#coding=utf-8
import tensorflow as tf
import wmodule
from .build import KEYPOINTS_HEAD
from object_detection2.modeling.onestage_heads.retinanet_outputs import *
from object_detection2.datadef import *
import object_detection2.od_toolkit as odtk

slim = tf.contrib.slim
@KEYPOINTS_HEAD.register()
class OpenPoseHead(wmodule.WChildModule):
    def __init__(self, num_keypoints,cfg, parent, *args, **kwargs):
        '''
        :param cfg:  only the child part
        :param parent:
        :param args:
        :param kwargs:
        '''
        super().__init__(cfg, *args, parent=parent, **kwargs)
        self.num_keypoints = num_keypoints
        self.normalizer_fn, self.norm_params = odtk.get_norm(self.cfg.NORM, is_training=self.is_training)
        self.activation_fn = odtk.get_activation_fn(self.cfg.ACTIVATION_FN)
        self.norm_scope_name = odtk.get_norm_scope_name(self.cfg.NORM)
        self.pred_paf_maps_outputs = []
        self.pred_conf_maps_outputs = []

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            pred_maps (list[Tensor]): #lvl tensors, each has shape (N, Hi, Wi,AxK).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
        """
        assert len(features)==1,f"Error features nr for open pose {len(features)}"
        cfg = self.cfg
        num_classes = self.num_keypoints
        num_l1_blocks = cfg.NUM_OPENPOSE_L1_BLOCKS
        num_l2_blocks = cfg.NUM_OPENPOSE_L2_BLOCKS
        num_units = cfg.NUM_OPENPOSE_UNITS
        assert num_l1_blocks>0,f"Error num l1 blocks for openpose {num_l1_blocks}"
        assert num_l2_blocks>0,f"Error num l1 blocks for openpose {num_l2_blocks}"
        self.pred_paf_maps_outputs= []
        self.pred_conf_maps_outputs= []
        feature = features[0]
        feature = slim.conv2d(feature, 256, [3, 3],
                          activation_fn=self.activation_fn,
                          normalizer_fn=self.normalizer_fn,
                          normalizer_params=self.norm_params,
                          biases_initializer=None if self.normalizer_fn is not None else tf.zeros_initializer(),
                          scope=f"conv2d_1")
        feature = slim.conv2d(feature, 128, [3, 3],
                              activation_fn=self.activation_fn,
                              normalizer_fn=self.normalizer_fn,
                              normalizer_params=self.norm_params,
                              biases_initializer=None if self.normalizer_fn is not None else tf.zeros_initializer(),
                              scope=f"conv2d_2")
        oc = len(self.cfg.POINTS_PAIRS)
        unit_channels = self.cfg.OPENPOSE_UNIT_CHANNELS
        net = feature
        for i in range(num_l1_blocks):
            with tf.variable_scope(f"Block{i}"):
                if i == 0:
                    u_c = unit_channels[0]
                    mid_c = 256
                else:
                    u_c = unit_channels[1]
                    mid_c = 512

                for i in range(num_units):
                    net = self.unit(net,channels=u_c,name=f"unit{i}")
                net = slim.conv2d(net, mid_c, [1, 1],
                                  activation_fn=self.activation_fn,
                                  normalizer_fn=self.normalizer_fn,
                                  normalizer_params=self.norm_params,
                                  biases_initializer=None if self.normalizer_fn is not None else tf.zeros_initializer(),
                                  scope=f"conv2d_1")
                net = slim.conv2d(net, oc, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=None if self.normalizer_fn is not None else tf.zeros_initializer(),
                                  scope=f"conv2d_2")
                self.pred_paf_maps_outputs.append(net)
                net = tf.concat([feature,net],axis=-1)

        last_l1_feature = self.pred_paf_maps_outputs[-1]
        oc = num_classes
        net = tf.concat([feature, last_l1_feature], axis=-1)
        for i in range(num_l2_blocks):
            if i == 0:
                u_c = unit_channels[0]
                mid_c = 256
            else:
                u_c = unit_channels[1]
                mid_c = 512

            if i>0:
                net = tf.concat([feature, last_l1_feature, net], axis=-1)
            with tf.variable_scope(f"Block{i+num_l1_blocks}"):
                for i in range(num_units):
                    net = self.unit(net,channels=u_c,name=f"unit{i}")
                net = slim.conv2d(net, mid_c, [1, 1],
                                  activation_fn=self.activation_fn,
                                  normalizer_fn=self.normalizer_fn,
                                  normalizer_params=self.norm_params,
                                  biases_initializer=None if self.normalizer_fn is not None else tf.zeros_initializer(),
                                  scope=f"conv2d_1")
                net = slim.conv2d(net, oc, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope=f"conv2d_2")
                self.pred_conf_maps_outputs.append(net)
        output = tf.concat([self.pred_conf_maps_outputs[-1],self.pred_paf_maps_outputs[-1]],axis=-1,name="output")
        return self.pred_paf_maps_outputs,self.pred_conf_maps_outputs,output

    def unit(self,net,channels=96,name=None):
        outputs = []
        with tf.variable_scope(name,default_name="unit"):
            for i in range(3):
                net = slim.conv2d(net, channels, [3, 3],
                                  activation_fn=self.activation_fn,
                                  normalizer_fn=self.normalizer_fn,
                                  normalizer_params=self.norm_params,
                                  biases_initializer=None if self.normalizer_fn is not None else tf.zeros_initializer(),
                                  scope=f"conv2d_{i}")
                outputs.append(net)
            return tf.concat(outputs,axis=-1)
