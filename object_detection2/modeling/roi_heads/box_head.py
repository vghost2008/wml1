import numpy as np
import tensorflow as tf
from thirdparty.registry import Registry
import wmodule
import wml_tfutils as wmlt
import wnnlayer as wnnl
import object_detection2.od_toolkit as odt
from collections import Iterable
from .build import ROI_BOX_HEAD_REGISTRY

slim = tf.contrib.slim

@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHead(wmodule.WChildModule):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg,*args,**kwargs):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__(cfg,*args,**kwargs)
        '''
        Detectron2默认没有使用任何normalizer
        但我在使用测试数据时发现，使用group_norm(默认参数)后性能会显著提升
        '''
        self.normalizer_fn = wnnl.group_norm


    def forward(self, x,scope="FastRCNNConvFCHead"):
        with tf.variable_scope(scope):
            cfg = self.cfg
            conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
            num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
            fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
            num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC

            assert num_conv + num_fc > 0

            for _ in range(num_conv):
                x = slim.conv2d(x,conv_dim,[3,3],activation_fn=tf.nn.relu,
                                normalizer_fn=None)
            if num_fc>0:
                if len(x.get_shape()) > 2:
                    shape = wmlt.combined_static_and_dynamic_shape(x)
                    dim = 1
                    for i in range(1,len(shape)):
                        dim = dim*shape[i]
                    x = tf.reshape(x,[shape[0],dim])
                for _ in range(num_fc):
                    x = slim.fully_connected(x,fc_dim,activation_fn=tf.nn.relu,
                                             normalizer_fn=self.normalizer_fn)

            return x


@ROI_BOX_HEAD_REGISTRY.register()
class SeparateFastRCNNConvFCHead(wmodule.WChildModule):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg,*args,**kwargs):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__(cfg,*args,**kwargs)
        self.normalizer_fn,self.norm_params = odt.get_norm(self.cfg.MODEL.ROI_BOX_HEAD.NORM,self.is_training)
        self.activation_fn = odt.get_activation_fn(self.cfg.MODEL.ROI_BOX_HEAD.ACTIVATION_FN)


    def forward(self, x,scope="FastRCNNConvFCHead"):
        with tf.variable_scope(scope):
            cfg = self.cfg
            conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
            num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
            fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
            num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC

            assert num_conv + num_fc > 0

            if not isinstance(x,tf.Tensor) and isinstance(x,Iterable):
                assert len(x) == 2,"error feature map length"
                cls_x = x[0]
                box_x = x[1]
            else:
                cls_x = x
                box_x = x

            with tf.variable_scope("ClassPredictionTower"):
                for _ in range(num_conv):
                    cls_x = slim.conv2d(cls_x,conv_dim,[3,3],
                                        activation_fn=self.activation_fn,
                                        normalizer_fn=self.normalizer_fn,
                                        normalizer_params=self.norm_params)
                if num_fc>0:
                    if len(cls_x.get_shape()) > 2:
                        shape = wmlt.combined_static_and_dynamic_shape(cls_x)
                        dim = 1
                        for i in range(1,len(shape)):
                            dim = dim*shape[i]
                        cls_x = tf.reshape(cls_x,[shape[0],dim])
                    for _ in range(num_fc):
                        cls_x = slim.fully_connected(cls_x,fc_dim,
                                                     activation_fn=self.activation_fn,
                                                     normalizer_fn=self.normalizer_fn,
                                                     normalizer_params=self.norm_params)
            with tf.variable_scope("BoxPredictionTower"):
                for _ in range(num_conv):
                    box_x = slim.conv2d(box_x,conv_dim,[3,3],
                                        activation_fn=self.activation_fn,
                                        normalizer_fn=self.normalizer_fn,
                                        normalizer_params=self.norm_params)
                if num_fc>0:
                    if len(box_x.get_shape()) > 2:
                        shape = wmlt.combined_static_and_dynamic_shape(box_x)
                        dim = 1
                        for i in range(1,len(shape)):
                            dim = dim*shape[i]
                        box_x = tf.reshape(box_x,[shape[0],dim])
                    for _ in range(num_fc):
                        box_x = slim.fully_connected(box_x,fc_dim,
                                                     activation_fn=self.activation_fn,
                                                     normalizer_fn=self.normalizer_fn,
                                                     normalizer_params=self.norm_params)

            return cls_x,box_x

@ROI_BOX_HEAD_REGISTRY.register()
class SeparateFastRCNNConvFCHeadV2(wmodule.WChildModule):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg,*args,**kwargs):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__(cfg,*args,**kwargs)
        self.normalizer_fn,self.norm_params = odt.get_norm(self.cfg.MODEL.ROI_BOX_HEAD.NORM,self.is_training)
        self.activation_fn = odt.get_activation_fn(self.cfg.MODEL.ROI_BOX_HEAD.ACTIVATION_FN)


    def forward(self, x,scope="FastRCNNConvFCHead",reuse=None):
        with tf.variable_scope(scope,reuse=reuse):
            cfg = self.cfg
            conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
            num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
            fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
            num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC

            assert num_conv + num_fc > 0

            if not isinstance(x,tf.Tensor) and isinstance(x,Iterable):
                assert len(x) == 2,"error feature map length"
                cls_x = x[0]
                box_x = x[1]
            else:
                cls_x = x
                box_x = x

            with tf.variable_scope("ClassPredictionTower"):
                if num_fc>0:
                    if len(cls_x.get_shape()) > 2:
                        shape = wmlt.combined_static_and_dynamic_shape(cls_x)
                        dim = 1
                        for i in range(1,len(shape)):
                            dim = dim*shape[i]
                        cls_x = tf.reshape(cls_x,[shape[0],dim])
                    for _ in range(num_fc):
                        cls_x = slim.fully_connected(cls_x,fc_dim,
                                                     activation_fn=self.activation_fn,
                                                     normalizer_fn=self.normalizer_fn,
                                                     normalizer_params=self.norm_params)
            with tf.variable_scope("BoxPredictionTower"):
                for _ in range(num_conv):
                    box_x = slim.conv2d(box_x,conv_dim,[3,3],
                                        activation_fn=self.activation_fn,
                                        normalizer_fn=self.normalizer_fn,
                                        normalizer_params=self.norm_params)

            return cls_x,box_x
