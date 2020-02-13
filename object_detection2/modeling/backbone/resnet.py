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
        with slim.arg_scope(resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=self.is_training):
                with tf.variable_scope("FeatureExtractor"):
                    _,end_points = resnet_v1_50(x['image'],output_stride=None)

        self.end_points = end_points

        for i in range(1,5):
            res[f"C{i+1}"] = end_points["FeatureExtractor/resnet_v1_50/"+f"block{i}"]

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
