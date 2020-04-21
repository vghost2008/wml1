#coding=utf-8
import tensorflow as tf
from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from thirdparty.nets.resnet_v1 import *
import collections

slim = tf.contrib.slim


class ResNet(Backbone):
    def __init__(self,cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)

    def forward(self, x):
        res = collections.OrderedDict()
        batch_norm_decay = self.cfg.MODEL.RESNETS.batch_norm_decay #0.999
        if not self.is_training:
            train_bn = False
        elif self.cfg.MODEL.RESNETS.FROZEN_BN:
            print("Frozen bn.")
            train_bn = False
        else:
            train_bn = True
        with slim.arg_scope(resnet_arg_scope(batch_norm_decay=batch_norm_decay,
                                             is_training=train_bn)):
            with tf.variable_scope("FeatureExtractor"):
                _,end_points = resnet_v1_50(x['image'],output_stride=None)

        self.end_points = end_points

        keys = ["conv1","block1/unit_2/bottleneck_v1","block1","block2","block4"] #block3,block4都是1/32
        keys2 = ["block1","block2","block3","block4"] #block3,block4都是1/32
        values2 = ["res1","res2","res3","res4"] #block3,block4都是1/32
        for i in range(1,6):
            res[f"C{i}"] = end_points["FeatureExtractor/resnet_v1_50/"+keys[i-1]]
        for i,k in enumerate(keys2):
            res[values2[i]] = end_points["FeatureExtractor/resnet_v1_50/"+keys2[i]]

        return res


@BACKBONE_REGISTRY.register()
def build_resnet_backbone(cfg, *args,**kwargs):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    return ResNet(cfg,*args,**kwargs)
