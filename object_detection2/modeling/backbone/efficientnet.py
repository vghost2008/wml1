#coding=utf-8
import tensorflow as tf
from .backbone import Backbone
from .build import BACKBONE_REGISTRY
import thirdparty.nets.efficientnet.efficientnet_builder as eb
import collections
import object_detection2.od_toolkit as odt
import wmodule

slim = tf.contrib.slim

class EfficientNet(Backbone):
    def __init__(self,cfg,*args,**kwargs):
        if cfg.MODEL.PREPROCESS != "m0v1":
            print("--------------------WARNING--------------------")
            print(f"Preprocess for efficientnet should be m0v1 not {cfg.MODEL.PREPROCESS}.")
            print("------------------END WARNING------------------")

        super().__init__(cfg,*args,**kwargs)
        self.efficient_net_type = cfg.MODEL.EFFICIENTNETS.TYPE
        self.efficient_net_name = f"efficientnet-b{self.efficient_net_type}"

    def forward(self, x):
        res = collections.OrderedDict()
        if not self.is_training:
            train_bn = False
        elif self.cfg.MODEL.EFFICIENTNETS.FROZEN_BN:
            print("Frozen bn.")
            train_bn = False
        else:
            train_bn = True
        features, end_points = eb.build_model_base(x['image'], self.efficient_net_name,
                                                       training=train_bn)
        self.end_points = end_points

        for i in range(1,6):
            res[f"C{i}"] = end_points[f'reduction_{i}']
        
        '''
        EfficientDet官方实现时通过一个conv+max_pool增加了一层输出，这里可以选择增加一层或两层，且使用stride下采样
        '''
        if self.cfg.MODEL.EFFICIENTNETS.MAKE_C6C7 != "":
            conv_nr = 1 if self.cfg.MODEL.EFFICIENTNETS.MAKE_C6C7=="C6" else 2
            lp6p7 = LastLevelP6P7(out_channels=256,cfg=self.cfg,parent=self,conv_nr=conv_nr)
            p6p7 = lp6p7(res["C5"])
            for i,net in enumerate(p6p7):
                res[f"C{6+i}"] = p6p7[i]
        return res

class LastLevelP6P7(wmodule.WChildModule):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, out_channels,cfg,conv_nr=1,*args,**kwargs):
        super().__init__(cfg=cfg,*args,**kwargs)
        self.normalizer_fn,self.norm_params = odt.get_norm(self.cfg.MODEL.EFFICIENTNETS.NORM,self.is_training)
        self.activation_fn = odt.get_activation_fn(self.cfg.MODEL.EFFICIENTNETS.ACTIVATION_FN)
        self.out_channels = out_channels
        self.conv_nr = conv_nr

    def forward(self, c5):
        with tf.variable_scope("EfficientNetLastLevel"):
            res = []
            last_feature = c5
            for i in range(self.conv_nr):
                last_feature = slim.conv2d(last_feature,self.out_channels,[3,3],stride=2,
                                            activation_fn=self.activation_fn,
                                            normalizer_fn=self.normalizer_fn,
                                            normalizer_params=self.norm_params,
                                            scope=f"conv{i+1}")
                res.append(last_feature)
        return res

@BACKBONE_REGISTRY.register()
def build_efficientnet_backbone(cfg, *args,**kwargs):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    return EfficientNet(cfg,*args,**kwargs)
