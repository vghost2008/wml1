#coding=utf-8
import tensorflow as tf
import wmodule
from .build import KEYPOINTS_HEAD
from object_detection2.modeling.onestage_heads.retinanet_outputs import *
from object_detection2.datadef import *
import object_detection2.od_toolkit as odtk
import wnnlayer as wnnl

slim = tf.contrib.slim

@KEYPOINTS_HEAD.register()
class HRNetPEHead(wmodule.WChildModule):
    def __init__(self, num_keypoints, cfg, parent, *args, **kwargs):
        '''
        :param cfg:  only the child part
        :param parent:
        :param args:
        :param kwargs:
        '''
        super().__init__(cfg, *args, parent=parent, **kwargs)
        self.num_keypoints = num_keypoints
        self.normalizer_fn, self.norm_params = odtk.get_norm(
            self.cfg.NORM, is_training=self.is_training)
        self.activation_fn = odtk.get_activation_fn(self.cfg.ACTIVATION_FN)
        self.norm_scope_name = odtk.get_norm_scope_name(self.cfg.NORM)
        

    def combine_backbone_outputs(self, xs, scope=None):
        layer_idxs = list(range(len(xs)))
        target_scale = layer_idxs[0]
        x = xs[0]
        batch_size, target_height, target_width, target_channel = x.shape
        with tf.variable_scope(scope, default_name="combin_backbone"):
            for downsample_scal in layer_idxs[1:]:
                y = xs[downsample_scal]
                y = slim.conv2d(y, target_channel, 1, 1, activation_fn=self.activation_fn,
                                  normalizer_fn=self.normalizer_fn,
                                  normalizer_params=self.norm_params)
                y = wnnl.upsample(y, scale_factor=2 ** (downsample_scal - target_scale)
                    , mode=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                x += y

        return x


    def final_layers_0(self, x, scope=None):
        target_channel = 2 * self.cfg.NUM_KEYPOINTS
        with tf.variable_scope(scope, default_name="final_layers_0"):
            x = slim.conv2d(x, target_channel, 1, 1, activation_fn=None,
                            normalizer_fn=None,
                            normalizer_params=None)
        return x


    def final_layers_1(self, x, scope=None):
        target_channel = self.cfg.NUM_KEYPOINTS
        with tf.variable_scope(scope, default_name="final_layers_1"):
            x = slim.conv2d(x, target_channel, 1, 1, activation_fn=None,
                            normalizer_fn=None,
                            normalizer_params=None)
        return x


    def basic_block(self, x, target_channel, scope=None):
        residual = x
        with tf.variable_scope(scope, default_name="basic_block"):
            out = slim.conv2d(x, target_channel, 3,
                              activation_fn=self.activation_fn,
                              normalizer_fn=self.normalizer_fn,
                              normalizer_params=self.norm_params)
            out = slim.conv2d(out, target_channel, 3,
                              activation_fn=None,
                              normalizer_fn=self.normalizer_fn,
                              normalizer_params=self.norm_params)
            out += residual
            out = self.activation_fn(out)

        return out


    def deconv_layers(self, x, target_channel, scope=None):
        with tf.variable_scope(scope, default_name="deconv_layers"):
            x = slim.convolution2d_transpose(x, target_channel,
                                             kernel_size=4, stride=2,
                                             activation_fn=self.activation_fn,
                                             normalizer_fn=self.normalizer_fn,
                                             normalizer_params=self.norm_params,
                                             padding="SAME")
            for i in range(4):
                x = self.basic_block(x, target_channel,scope=f"block{i}")
        return x

    def head(self, x, scope=None):
        output = []
        with tf.variable_scope(scope, default_name="head"):
            x = self.combine_backbone_outputs(x, scope="combine_backbone")
            y = self.final_layers_0(x, scope=f"final_layers_0")
            output.append(y)

            x = tf.concat([x, y], -1)
            x = self.deconv_layers(x, 32, "deconv_layers")
            y = self.final_layers_1(x, scope="final_layers_1")
            output.append(y)

        return output

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): HRNet feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            pred_maps (list[Tensor]): #lvl tensors, the first have shape (N, Hi, Wi,NUM_KEYPOINTS*2), predicts the
                keypoints location and tags, the second tensor have shape (N,Hi,Wi,NUM_KEYPOINTS), predicts the keypoints
                location in higher resolution
        """
        return self.head(features, "head")
