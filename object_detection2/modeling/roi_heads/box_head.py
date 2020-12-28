import numpy as np
import tensorflow as tf
from thirdparty.registry import Registry
import object_detection2.bboxes as odb
import wsummary
import wmodule
import wml_tfutils as wmlt
import wnnlayer as wnnl
import object_detection2.od_toolkit as odt
from collections import Iterable
from .build import ROI_BOX_HEAD_REGISTRY
import basic_tftools as btf
from object_detection2.modeling.matcher import Matcher
from object_detection2.standard_names import *
from object_detection2.datadef import *

slim = tf.contrib.slim
class BoxesForwardType:
    BBOXES = 1
    CLASSES = 2
    IOUS = 4
    ALL = 15

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
            if cfg.MODEL.ROI_BOX_HEAD.FC_WEIGHT_DECAY > 0.0:
               fc_weights_regularizer = slim.l2_regularizer(cfg.MODEL.ROI_BOX_HEAD.FC_WEIGHT_DECAY)
            else:
               fc_weights_regularizer = None
            if cfg.MODEL.ROI_BOX_HEAD.CONV_WEIGHT_DECAY > 0.0:
               conv_weights_regularizer = slim.l2_regularizer(cfg.MODEL.ROI_BOX_HEAD.CONV_WEIGHT_DECAY)
            else:
               conv_weights_regularizer = None

            with tf.variable_scope("ClassPredictionTower"):
                for _ in range(num_conv):
                    cls_x = slim.conv2d(cls_x,conv_dim,[3,3],
                                        activation_fn=self.activation_fn,
                                        normalizer_fn=self.normalizer_fn,
                                        normalizer_params=self.norm_params,
                                        weights_regularizer=conv_weights_regularizer,
                                        padding = 'SAME')


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
                                                     weights_regularizer=fc_weights_regularizer,
                                                     normalizer_params=self.norm_params)

            with tf.variable_scope("BoxPredictionTower"):
                for _ in range(num_conv):
                    box_x = slim.conv2d(box_x,conv_dim,[3,3],
                                        activation_fn=self.activation_fn,
                                        normalizer_fn=self.normalizer_fn,
                                        normalizer_params=self.norm_params,
                                        weights_regularizer=conv_weights_regularizer,
                                        padding = 'SAME')

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
                                                     weights_regularizer=fc_weights_regularizer,
                                                     normalizer_params=self.norm_params)

            return cls_x,box_x

@ROI_BOX_HEAD_REGISTRY.register()
class SeparateFastRCNNConvFCHeadV2(wmodule.WChildModule):
    """
    Rethinking Classification and Localization for Object Detection
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
                assert len(x) >= 2,"error feature map length"
                cls_x = x[0]
                box_x = x[1]
                if len(x)>=3:
                    iou_x = x[2]
                else:
                    iou_x = x[1]
            else:
                cls_x = x
                box_x = x
                iou_x = x

            with tf.variable_scope("ClassPredictionTower"):
                if num_fc>0:
                    if len(cls_x.get_shape()) > 2:
                        shape = wmlt.combined_static_and_dynamic_shape(cls_x)
                        dim = 1
                        for i in range(1,len(shape)):
                            dim = dim*shape[i]
                        cls_x = tf.reshape(cls_x,[shape[0],dim])
                    if cfg.MODEL.ROI_BOX_HEAD.FC_WEIGHT_DECAY > 0.0:
                        weights_regularizer = slim.l2_regularizer(cfg.MODEL.ROI_BOX_HEAD.FC_WEIGHT_DECAY)
                    else:
                        weights_regularizer = None
                    for _ in range(num_fc):
                        cls_x = slim.fully_connected(cls_x,fc_dim,
                                                     activation_fn=self.activation_fn,
                                                     normalizer_fn=self.normalizer_fn,
                                                     normalizer_params=self.norm_params,
                                                     weights_regularizer=weights_regularizer)

                    if cfg.MODEL.ROI_BOX_HEAD.ENABLE_CLS_DROPBLOCK and self.is_training:
                        keep_prob = wnnl.get_dropblock_keep_prob(tf.train.get_or_create_global_step(),self.cfg.SOLVER.STEPS[-1],
                                                                 max_keep_prob=self.cfg.MODEL.ROI_BOX_HEAD.CLS_KEEP_PROB)
                        if self.cfg.GLOBAL.SUMMARY_LEVEL <= SummaryLevel.DEBUG:
                            tf.summary.scalar(name="box_head_cls_keep_prob",tensor=keep_prob)
                        cls_x = slim.dropout(cls_x, keep_prob=keep_prob,is_training=self.is_training)

            with tf.variable_scope("BoxPredictionTower"):
                if cfg.MODEL.ROI_BOX_HEAD.CONV_WEIGHT_DECAY > 0.0:
                    weights_regularizer = slim.l2_regularizer(cfg.MODEL.ROI_BOX_HEAD.CONV_WEIGHT_DECAY)
                else:
                    weights_regularizer = None
                for _ in range(num_conv):
                    box_x = slim.conv2d(box_x,conv_dim,[3,3],
                                        activation_fn=self.activation_fn,
                                        normalizer_fn=self.normalizer_fn,
                                        normalizer_params=self.norm_params,
                                        weights_regularizer=weights_regularizer)

                if cfg.MODEL.ROI_BOX_HEAD.ENABLE_BOX_DROPBLOCK and self.is_training:
                    keep_prob = wnnl.get_dropblock_keep_prob(tf.train.get_or_create_global_step(),self.cfg.SOLVER.STEPS[-1],
                                                             max_keep_prob=self.cfg.MODEL.ROI_BOX_HEAD.BOX_KEEP_PROB)
                    if self.cfg.GLOBAL.SUMMARY_LEVEL <= SummaryLevel.DEBUG:
                        tf.summary.scalar(name="box_head_box_keep_prob",tensor=keep_prob)
                    box_x = slim.dropout(box_x, keep_prob=keep_prob,is_training=self.is_training)

            if cfg.MODEL.ROI_HEADS.PRED_IOU:
                iou_num_conv = cfg.MODEL.ROI_BOX_HEAD.IOU_NUM_CONV
                iou_num_fc = cfg.MODEL.ROI_BOX_HEAD.IOU_NUM_FC
                with tf.variable_scope("BoxIOUPredictionTower"):
                    for _ in range(iou_num_conv):
                        iou_x = slim.conv2d(iou_x,conv_dim,[3,3],
                                            activation_fn=self.activation_fn,
                                            normalizer_fn=self.normalizer_fn,
                                            normalizer_params=self.norm_params)
                    if iou_num_fc>0:
                        if len(iou_x.get_shape()) > 2:
                            shape = wmlt.combined_static_and_dynamic_shape(iou_x)
                            dim = 1
                            for i in range(1,len(shape)):
                                dim = dim*shape[i]
                            iou_x = tf.reshape(iou_x,[shape[0],dim])
                        for _ in range(iou_num_fc):
                            iou_x = slim.fully_connected(iou_x,fc_dim,
                                                         activation_fn=self.activation_fn,
                                                         normalizer_fn=self.normalizer_fn,
                                                         normalizer_params=self.norm_params)

            if cfg.MODEL.ROI_HEADS.PRED_IOU:
                return cls_x,box_x,iou_x
            else:
                return cls_x,box_x

@ROI_BOX_HEAD_REGISTRY.register()
class SeparateFastRCNNConvFCHeadV3(wmodule.WChildModule):
    """
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
                lconv_dim = conv_dim
                for _ in range(num_conv):
                    lconv_dim += 64
                    cls_x = slim.conv2d(cls_x,lconv_dim,[3,3],
                                        activation_fn=self.activation_fn,
                                        normalizer_fn=self.normalizer_fn,
                                        normalizer_params=self.norm_params,
                                        padding = 'VALID')


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
                lconv_dim = conv_dim
                for _ in range(num_conv):
                    lconv_dim += 64
                    box_x = slim.conv2d(box_x,lconv_dim,[3,3],
                                        activation_fn=self.activation_fn,
                                        normalizer_fn=self.normalizer_fn,
                                        normalizer_params=self.norm_params,
                                        padding = 'VALID')

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
class SeparateFastRCNNConvFCHeadV4(wmodule.WChildModule):
    """
    Rethinking Classification and Localization for Object Detection
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


    def forward(self, x,scope="FastRCNNConvFCHead",reuse=None,fwd_type=BoxesForwardType.ALL):
        with tf.variable_scope(scope,reuse=reuse):
            cfg = self.cfg
            conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
            num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
            fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
            num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC

            assert num_conv + num_fc > 0

            if not isinstance(x,tf.Tensor) and isinstance(x,Iterable):
                assert len(x) >= 2,"error feature map length"
                cls_x = x[0]
                box_x = x[1]
                if cfg.MODEL.ROI_HEADS.PRED_IOU:
                    if len(x) == 3:
                        iou_x = x[2]
                    else:
                        iou_x = box_x
            else:
                cls_x = x
                box_x = x
                iou_x = x

            if fwd_type&BoxesForwardType.CLASSES:
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
            if fwd_type&BoxesForwardType.BBOXES:
                with tf.variable_scope("BoxPredictionTower"):
                    nets = []
                    for _ in range(num_conv):
                        box_x = slim.conv2d(box_x,conv_dim,[3,3],
                                            activation_fn=self.activation_fn,
                                            normalizer_fn=self.normalizer_fn,
                                            normalizer_params=self.norm_params)
                        nets.append(box_x)

            if cfg.MODEL.ROI_HEADS.PRED_IOU and ( fwd_type&BoxesForwardType.IOUS or fwd_type&BoxesForwardType.BBOXES):
                with tf.variable_scope("BoxIOUPredictionTower",reuse=tf.AUTO_REUSE):
                    if num_fc > 0:
                        if len(iou_x.get_shape()) > 2:
                            shape = wmlt.combined_static_and_dynamic_shape(iou_x)
                            dim = 1
                            for i in range(1, len(shape)):
                                dim = dim * shape[i]
                            iou_x = tf.reshape(iou_x, [shape[0], dim])
                        for _ in range(num_fc):
                            iou_x = slim.fully_connected(iou_x, fc_dim,
                                                         activation_fn=self.activation_fn,
                                                         normalizer_fn=self.normalizer_fn,
                                                         normalizer_params=self.norm_params)
                        box_channel = btf.channel(box_x)
                        att = slim.fully_connected(iou_x,box_channel,
                                                   activation_fn=tf.nn.sigmoid,
                                                   normalizer_fn=None)
                        att = tf.expand_dims(att, axis=1)
                        att = tf.expand_dims(att, axis=1)
                        wsummary.histogram_or_scalar(att,"iou_box_att")
                        box_x = box_x*att

            if cfg.MODEL.ROI_HEADS.PRED_IOU:
                return cls_x,box_x,iou_x
            else:
                return cls_x,box_x

@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvHead(wmodule.WChildModule):

    def __init__(self, cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)
        self.normalizer_fn,self.norm_params = odt.get_norm(self.cfg.MODEL.ROI_BOX_HEAD.NORM,self.is_training)
        self.activation_fn = odt.get_activation_fn(self.cfg.MODEL.ROI_BOX_HEAD.ACTIVATION_FN)


    def forward(self, x,scope="FastRCNNConvHead",reuse=None):
        with tf.variable_scope(scope,reuse=reuse):
            cfg = self.cfg
            conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
            num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV

            for _ in range(num_conv):
                x = slim.conv2d(x,conv_dim,[3,3],
                                activation_fn=self.activation_fn,
                                normalizer_params=self.norm_params,
                                normalizer_fn=self.normalizer_fn)
            cls_x = x
            box_x = x
            msk_x = x
            cls_x = slim.conv2d(cls_x,256,[3,3],
                                activation_fn=self.activation_fn,
                                normalizer_fn=self.normalizer_fn,
                                normalizer_params=self.norm_params,
                                padding = 'SAME',
                                scope="cls")


            box_x = slim.conv2d(box_x,256,[3,3],
                                activation_fn=self.activation_fn,
                                normalizer_fn=self.normalizer_fn,
                                normalizer_params=self.norm_params,
                                padding = 'VALID',
                                scope="box")
            if cfg.MODEL.MASK_ON:
                msk_x = slim.conv2d(msk_x, 256, [3, 3],
                                    activation_fn=self.activation_fn,
                                    normalizer_fn=self.normalizer_fn,
                                    normalizer_params=self.norm_params,
                                    padding='SAME',
                                    scope="mask")
                return cls_x,box_x,msk_x
            else:
                return cls_x,box_x

@ROI_BOX_HEAD_REGISTRY.register()
class SeparateFastRCNNConvFCHeadV5(wmodule.WChildModule):
    """
    Rethinking Classification and Localization for Object Detection
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

            assert len(x) == 2,"error feature map length"
            cls_x = x[0]
            box_x = wnnl.non_local_blockv3(x[0],x[1],x[1],inner_dims_multiplier=[1,1,1],
                                           normalizer_params=self.norm_params,
                                           normalizer_fn=self.normalizer_fn,
                                           activation_fn=self.activation_fn)
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
                nets = []
                for _ in range(num_conv):
                    box_x = slim.conv2d(box_x,conv_dim,[3,3],
                                        activation_fn=self.activation_fn,
                                        normalizer_fn=self.normalizer_fn,
                                        normalizer_params=self.norm_params)
                    nets.append(box_x)

            if cfg.MODEL.ROI_HEADS.PRED_IOU:
                with tf.variable_scope("BoxIOUPredictionTower"):
                    net = nets[-2]
                    iou_x = slim.conv2d(net,conv_dim,[3,3],
                                        activation_fn=self.activation_fn,
                                        normalizer_fn=self.normalizer_fn,
                                        normalizer_params=self.norm_params)

            if cfg.MODEL.ROI_HEADS.PRED_IOU:
                return cls_x,box_x,iou_x
            else:
                return cls_x,box_x

@ROI_BOX_HEAD_REGISTRY.register()
class SeparateFastRCNNConvFCHeadV6(wmodule.WChildModule):
    """
    Rethinking Classification and Localization for Object Detection
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

            assert len(x) == 2,"error feature map length"
            cls_x = x[0]
            box_x = wnnl.non_local_blockv3(x[1],x[0],x[0],inner_dims_multiplier=[1,1,1],
                                           normalizer_params=self.norm_params,
                                           normalizer_fn=self.normalizer_fn,
                                           activation_fn=self.activation_fn)
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
                nets = []
                for _ in range(num_conv):
                    box_x = slim.conv2d(box_x,conv_dim,[3,3],
                                        activation_fn=self.activation_fn,
                                        normalizer_fn=self.normalizer_fn,
                                        normalizer_params=self.norm_params)
                    nets.append(box_x)

            if cfg.MODEL.ROI_HEADS.PRED_IOU:
                with tf.variable_scope("BoxIOUPredictionTower"):
                    net = nets[-2]
                    iou_x = slim.conv2d(net,conv_dim,[3,3],
                                        activation_fn=self.activation_fn,
                                        normalizer_fn=self.normalizer_fn,
                                        normalizer_params=self.norm_params)

            if cfg.MODEL.ROI_HEADS.PRED_IOU:
                return cls_x,box_x,iou_x
            else:
                return cls_x,box_x

@ROI_BOX_HEAD_REGISTRY.register()
class SeparateFastRCNNConvFCHeadV7(wmodule.WChildModule):
    """
    Rethinking Classification and Localization for Object Detection
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

            assert len(x) == 2,"error feature map length"
            with tf.variable_scope("Attenation"):
                _,S0,_,C = wmlt.combined_static_and_dynamic_shape(x[0])
                _,S1,_,_ = wmlt.combined_static_and_dynamic_shape(x[1])
                E_nr = (S1-S0)//2
                data0 = x[1][:,0:E_nr:,E_nr:E_nr+S0,:]
                data1 = x[1][:,E_nr:E_nr+S0,S1-E_nr:,:]
                data2 = x[1][:,S1-E_nr:,E_nr:E_nr+S0,:]
                data3 = x[1][:,E_nr:E_nr+S0,0:E_nr,:]
                def data_trans(_data):
                    i,data = _data
                    data = wnnl.non_local_blockv3(x[0],data,data,inner_dims_multiplier=[2,2,2],
                                               normalizer_params=self.norm_params,
                                               normalizer_fn=self.normalizer_fn,
                                               activation_fn=self.activation_fn,
                                               weighed_sum=False,skip_connect=False,
                                               scope=f"trans_data{i}")
                    return data
                datas = [data0,data1,data2,data3]
                datas = list(map(data_trans,enumerate(datas)))
                datas = tf.concat(datas,axis=-1)
                datas = tf.reduce_mean(datas,axis=[1,2],keepdims=False)
                datas = slim.fully_connected(datas,C, activation_fn=tf.nn.relu, normalizer_fn=None)
                att = slim.fully_connected(datas,C, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                att = tf.expand_dims(att,axis=1)
                att = tf.expand_dims(att,axis=1)

            x_data = x[0]*att


            cls_x = x_data
            box_x = x_data
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
                nets = []
                for _ in range(num_conv):
                    box_x = slim.conv2d(box_x,conv_dim,[3,3],
                                        activation_fn=self.activation_fn,
                                        normalizer_fn=self.normalizer_fn,
                                        normalizer_params=self.norm_params)
                    nets.append(box_x)

            if cfg.MODEL.ROI_HEADS.PRED_IOU:
                with tf.variable_scope("BoxIOUPredictionTower"):
                    net = nets[-2]
                    iou_x = slim.conv2d(net,conv_dim,[3,3],
                                        activation_fn=self.activation_fn,
                                        normalizer_fn=self.normalizer_fn,
                                        normalizer_params=self.norm_params)

            if cfg.MODEL.ROI_HEADS.PRED_IOU:
                return cls_x,box_x,iou_x
            else:
                return cls_x,box_x
            
@ROI_BOX_HEAD_REGISTRY.register()
class SeparateFastRCNNConvFCHeadV8(wmodule.WChildModule):
    """
    Rethinking Classification and Localization for Object Detection
    """
    MAX_IOU = 0.9
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
        self.iou_threshold = 0.5
        self.head_nr = 4
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES     #不包含背景
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.box_dim = 4

    def forward_branch(self,cls_x,box_x,branch):
        cfg = self.cfg

        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC

        assert num_conv + num_fc > 0

        with tf.variable_scope(f"ClassPredictionTower{branch}"):
            if num_fc > 0:
                if len(cls_x.get_shape()) > 2:
                    shape = wmlt.combined_static_and_dynamic_shape(cls_x)
                    dim = 1
                    for i in range(1, len(shape)):
                        dim = dim * shape[i]
                    cls_x = tf.reshape(cls_x, [shape[0], dim])
                for _ in range(num_fc):
                    cls_x = slim.fully_connected(cls_x, fc_dim,
                                                 activation_fn=self.activation_fn,
                                                 normalizer_fn=self.normalizer_fn,
                                                 normalizer_params=self.norm_params)
        with tf.variable_scope(f"BoxPredictionTower{branch}"):
            for _ in range(num_conv):
                box_x = slim.conv2d(box_x, conv_dim, [3, 3],
                                    activation_fn=self.activation_fn,
                                    normalizer_fn=self.normalizer_fn,
                                    normalizer_params=self.norm_params)

        return cls_x,box_x

    '''def forward_with_ious(self,cls_x,box_x,ious):
        cls_x_datas = []
        box_x_datas = []
        index = tf.nn.relu(ious-self.iou_threshold)*self.head_nr/(1-self.iou_threshold)
        for i in range(self.head_nr):
            data = self.forward_branch(cls_x,box_x,i)
            cls_x_datas.append(data[0])
            box_x_datas.append(data[1])

        wsummary.histogram_or_scalar(index,"head_index")
        index = tf.cast(index,tf.int32)
        index = tf.clip_by_value(index,clip_value_min=0,clip_value_max=self.head_nr-1)
        cls_x_datas = tf.stack(cls_x_datas,axis=1)
        cls_x = wmlt.batch_gather(cls_x_datas,index)
        box_x_datas = tf.stack(box_x_datas,axis=1)
        box_x = wmlt.batch_gather(box_x_datas,index)

        return cls_x,box_x'''

    def trans(self,net):
        if len(net.get_shape()) > 2:
            shape = wmlt.combined_static_and_dynamic_shape(net)
            dim = 1
            for x in shape[1:]:
                dim *= x
            return tf.reshape(net, [shape[0], dim])
        else:
            return net

    def forward_with_ious(self,cls_x,box_x,ious):
        foreground_num_classes = self.num_classes
        num_bbox_reg_classes = 1 if self.cls_agnostic_bbox_reg else foreground_num_classes
        
        cls_x_datas = []
        box_x_datas = []
        index = tf.nn.relu(ious-self.iou_threshold)*self.head_nr/(self.MAX_IOU-self.iou_threshold)
        wsummary.histogram_or_scalar(index,"head_index")
        index = tf.cast(index,tf.int32)
        index = tf.clip_by_value(index,clip_value_min=0,clip_value_max=self.head_nr-1)
        data_indexs = []
        B = btf.batch_size(cls_x)
        data_raw_indexs = tf.range(B,dtype=tf.int32)
        for i in range(self.head_nr):
            mask = tf.equal(index,i)
            data_indexs.append(tf.boolean_mask(data_raw_indexs,mask))
            t_cls_x = tf.boolean_mask(cls_x,mask)
            t_box_x = tf.boolean_mask(box_x,mask)
            data = self.forward_branch(t_cls_x,t_box_x,i)
            cls_x_datas.append(data[0])
            box_x_datas.append(data[1])

        cls_x_datas = tf.concat(cls_x_datas,axis=0)
        cls_x_datas = self.trans(cls_x_datas)
        box_x_datas = tf.concat(box_x_datas,axis=0)
        box_x_datas = self.trans(box_x_datas)
        with tf.variable_scope("BoxPredictor"):
            cls_x_datas = slim.fully_connected(cls_x_datas, self.num_classes + 1, activation_fn=None,
                                               normalizer_fn=None, scope="cls_score")
            box_x_datas = slim.fully_connected(box_x_datas, self.box_dim * num_bbox_reg_classes, activation_fn=None,
                                               normalizer_fn=None, scope="bbox_pred")
        data_indexs = tf.concat(data_indexs,axis=0)
        data_indexs = tf.reshape(data_indexs,[B,1])
        
        shape = wmlt.combined_static_and_dynamic_shape(cls_x_datas)
        shape[0] = B
        cls_x = tf.scatter_nd(data_indexs,cls_x_datas,shape)
        shape = wmlt.combined_static_and_dynamic_shape(box_x_datas)
        shape[0] = B
        box_x = tf.scatter_nd(data_indexs,box_x_datas,shape)

        return cls_x,box_x


    def forward(self, x,scope="FastRCNNConvFCHead",reuse=None):
        with tf.variable_scope(scope,reuse=reuse):
            cfg = self.cfg
            conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
            num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
            fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
            num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC

            assert num_conv + num_fc > 0

            if not isinstance(x,tf.Tensor) and isinstance(x,Iterable):
                assert len(x) >= 2,"error feature map length"
                cls_x = x[0]
                box_x = x[1]
                if len(x)>=3:
                    iou_x = x[2]
                else:
                    iou_x = x[1]
            else:
                cls_x = x
                box_x = x
                iou_x = x
            if cfg.MODEL.ROI_HEADS.PRED_IOU:
                iou_num_conv = cfg.MODEL.ROI_BOX_HEAD.IOU_NUM_CONV
                iou_num_fc = cfg.MODEL.ROI_BOX_HEAD.IOU_NUM_FC
                with tf.variable_scope("BoxIOUPredictionTower"):
                    for _ in range(iou_num_conv):
                        iou_x = slim.conv2d(iou_x,conv_dim,[3,3],
                                            activation_fn=self.activation_fn,
                                            normalizer_fn=self.normalizer_fn,
                                            normalizer_params=self.norm_params)
                    if iou_num_fc>0:
                        if len(iou_x.get_shape()) > 2:
                            shape = wmlt.combined_static_and_dynamic_shape(iou_x)
                            dim = 1
                            for i in range(1,len(shape)):
                                dim = dim*shape[i]
                            iou_x = tf.reshape(iou_x,[shape[0],dim])
                        for _ in range(iou_num_fc):
                            iou_x = slim.fully_connected(iou_x,fc_dim,
                                                         activation_fn=self.activation_fn,
                                                         normalizer_fn=self.normalizer_fn,
                                                         normalizer_params=self.norm_params)
                iou_x = self.trans(iou_x)
                with tf.variable_scope("BoxPredictor"):
                    iou_x = slim.fully_connected(iou_x, 1,
                                                  activation_fn=None,
                                                  normalizer_fn=None,
                                                  scope="iou_pred")

            with tf.name_scope("get_ious"):
                if self.is_training:
                    def fn0():
                        matcher = Matcher(
                            [1e-3],
                            allow_low_quality_matches=False,
                            cfg=self.cfg,
                            parent=self
                        )
                        mh_res0 = matcher(self.parent.t_proposal_boxes,
                                      self.parent.batched_inputs[GT_BOXES],
                                      self.parent.batched_inputs[GT_LABELS],
                                      self.parent.batched_inputs[GT_LENGTH])
                        ious0 = tf.reshape(mh_res0[1],[-1]) #proposal_boxes与gt boxes的 iou
                        return ious0

                    def fn1():
                        ious1 = tf.nn.sigmoid(iou_x)
                        ious1 = tf.reshape(ious1,[-1])
                        return ious1

                    use_gt_probability = wnnl.get_dropblock_keep_prob(tf.train.get_or_create_global_step(),self.cfg.SOLVER.STEPS[-1],
                                                                             max_keep_prob=0.01)
                    tf.summary.scalar(name="use_gt_ious_probability",tensor=use_gt_probability)
                    p = tf.random_uniform(())
                    ious = tf.cond(tf.less(p,use_gt_probability),fn0,fn1)
                else:
                    ious = tf.nn.sigmoid(iou_x)
                    ious = tf.reshape(ious,[-1])

            cls_x,box_x = self.forward_with_ious(cls_x,box_x,ious)

            if cfg.MODEL.ROI_HEADS.PRED_IOU:
                return cls_x,box_x,iou_x
            else:
                return cls_x,box_x

@ROI_BOX_HEAD_REGISTRY.register()
class SeparateFastRCNNConvFCHeadV9(wmodule.WChildModule):
    """
    Rethinking Classification and Localization for Object Detection
    """

    def __init__(self, cfg, *args, **kwargs):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__(cfg, *args, **kwargs)
        self.normalizer_fn, self.norm_params = odt.get_norm(self.cfg.MODEL.ROI_BOX_HEAD.NORM, self.is_training)
        self.activation_fn = odt.get_activation_fn(self.cfg.MODEL.ROI_BOX_HEAD.ACTIVATION_FN)
        self.size_threshold = 64
        self.head_nr = 4
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES  # 不包含背景
        self.cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.box_dim = 4

    def forward_branch(self, cls_x, box_x, branch):
        cfg = self.cfg

        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC

        assert num_conv + num_fc > 0

        with tf.variable_scope(f"ClassPredictionTower{branch}"):
            if num_fc > 0:
                if len(cls_x.get_shape()) > 2:
                    shape = wmlt.combined_static_and_dynamic_shape(cls_x)
                    dim = 1
                    for i in range(1, len(shape)):
                        dim = dim * shape[i]
                    cls_x = tf.reshape(cls_x, [shape[0], dim])
                for _ in range(num_fc):
                    cls_x = slim.fully_connected(cls_x, fc_dim,
                                                 activation_fn=self.activation_fn,
                                                 normalizer_fn=self.normalizer_fn,
                                                 normalizer_params=self.norm_params)
        with tf.variable_scope(f"BoxPredictionTower{branch}"):
            for _ in range(num_conv):
                box_x = slim.conv2d(box_x, conv_dim, [3, 3],
                                    activation_fn=self.activation_fn,
                                    normalizer_fn=self.normalizer_fn,
                                    normalizer_params=self.norm_params)

        return cls_x, box_x

    def trans(self, net):
        if len(net.get_shape()) > 2:
            shape = wmlt.combined_static_and_dynamic_shape(net)
            dim = 1
            for x in shape[1:]:
                dim *= x
            return tf.reshape(net, [shape[0], dim])
        else:
            return net

    def forward_with_size(self, cls_x, box_x, size):
        foreground_num_classes = self.num_classes
        num_bbox_reg_classes = 1 if self.cls_agnostic_bbox_reg else foreground_num_classes

        cls_x_datas = []
        box_x_datas = []
        index = tf.nn.relu(size) * self.head_nr /self.size_threshold
        wsummary.histogram_or_scalar(index, "head_index")
        index = tf.cast(index, tf.int32)
        index = tf.clip_by_value(index, clip_value_min=0, clip_value_max=self.head_nr - 1)
        data_indexs = []
        B = btf.batch_size(cls_x)
        data_raw_indexs = tf.range(B, dtype=tf.int32)
        for i in range(self.head_nr):
            mask = tf.equal(index, i)
            data_indexs.append(tf.boolean_mask(data_raw_indexs, mask))
            t_cls_x = tf.boolean_mask(cls_x, mask)
            t_box_x = tf.boolean_mask(box_x, mask)
            data = self.forward_branch(t_cls_x, t_box_x, i)
            cls_x_datas.append(data[0])
            box_x_datas.append(data[1])

        cls_x_datas = tf.concat(cls_x_datas, axis=0)
        cls_x_datas = self.trans(cls_x_datas)
        box_x_datas = tf.concat(box_x_datas, axis=0)
        box_x_datas = self.trans(box_x_datas)
        with tf.variable_scope("BoxPredictor"):
            cls_x_datas = slim.fully_connected(cls_x_datas, self.num_classes + 1, activation_fn=None,
                                               normalizer_fn=None, scope="cls_score")
            box_x_datas = slim.fully_connected(box_x_datas, self.box_dim * num_bbox_reg_classes, activation_fn=None,
                                               normalizer_fn=None, scope="bbox_pred")
        data_indexs = tf.concat(data_indexs, axis=0)
        data_indexs = tf.reshape(data_indexs, [B, 1])

        shape = wmlt.combined_static_and_dynamic_shape(cls_x_datas)
        shape[0] = B
        cls_x = tf.scatter_nd(data_indexs, cls_x_datas, shape)
        shape = wmlt.combined_static_and_dynamic_shape(box_x_datas)
        shape[0] = B
        box_x = tf.scatter_nd(data_indexs, box_x_datas, shape)

        return cls_x, box_x

    def forward(self, x, scope="FastRCNNConvFCHeadV9", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            cfg = self.cfg
            if not isinstance(x, tf.Tensor) and isinstance(x, Iterable):
                assert len(x) >= 2, "error feature map length"
                cls_x = x[0]
                box_x = x[1]
                if len(x) >= 3:
                    iou_x = x[2]
                else:
                    iou_x = x[1]
            else:
                cls_x = x
                box_x = x
                iou_x = x
            with tf.name_scope("get_size"):
                img_size = wmlt.combined_static_and_dynamic_shape(self.parent.batched_inputs[IMAGE])
                p_bboxes = odb.tfrelative_boxes_to_absolutely_boxes(self.parent.t_proposal_boxes,width=img_size[2],height=img_size[1])
                bboxes_size = tf.sqrt(odb.box_area(p_bboxes))
                bboxes_size = tf.reshape(bboxes_size,[-1])

            cls_x, box_x = self.forward_with_size(cls_x, box_x, bboxes_size)

            if cfg.MODEL.ROI_HEADS.PRED_IOU:
                return cls_x, box_x, iou_x
            else:
                return cls_x, box_x
