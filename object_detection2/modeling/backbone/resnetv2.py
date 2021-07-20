#coding=utf-8
import tensorflow as tf
from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from thirdparty.nets.resnet_v2 import *
import collections
import wmodule
import object_detection2.od_toolkit as odt

slim = tf.contrib.slim


class ResNetV2(Backbone):
    def __init__(self,cfg,*args,**kwargs):
        if cfg.MODEL.PREPROCESS != "subimagenetmean":
            print("--------------------WARNING--------------------")
            print(f"Preprocess for resnet should be subimagenetmean not {cfg.MODEL.PREPROCESS}.")
            print("------------------END WARNING------------------")
        super().__init__(cfg,*args,**kwargs)
        self.normalizer_fn, self.norm_params = odt.get_norm(self.cfg.MODEL.RESNETS.NORM, self.is_training)
        self.activation_fn = odt.get_activation_fn(self.cfg.MODEL.RESNETS.ACTIVATION_FN)
        self.out_channels = cfg.MODEL.RESNETS.OUT_CHANNELS

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
            if self.cfg.MODEL.RESNETS.DEPTH == 101:
                print("ResNet-100")
                _,end_points = resnet_v2_101(x['image'],output_stride=None,block4_stride=2,is_training=train_bn)
            elif self.cfg.MODEL.RESNETS.DEPTH == 152:
                print("ResNet-150")
                _, end_points = resnet_v2_152(x['image'], output_stride=None,is_training=train_bn)
            elif self.cfg.MODEL.RESNETS.DEPTH == 200:
                print("ResNet-150")
                _, end_points = resnet_v2_200(x['image'], output_stride=None,is_training=train_bn)
            else:
                print("ResNet-50")
                _,end_points = resnet_v2_50(x['image'],output_stride=None,is_training=train_bn)

        self.end_points = end_points

        if self.cfg.MODEL.RESNETS.DEPTH>50: 
            keys = ["conv1","block1/unit_2/bottleneck_v2","block1","block2","block3","block4"] #block3,block4都是1/32
        else:
            keys = ["conv1","block1/unit_2/bottleneck_v2","block1","block2","block4"] 
        keys2 = ["block1","block2","block3","block4"] #block3,block4都是1/32
        values2 = ["res1","res2","res3","res4"] #block3,block4都是1/32
        for i in range(1,len(keys)+1):
            res[f"C{i}"] = end_points[f"resnet_v2_{self.cfg.MODEL.RESNETS.DEPTH}/"+keys[i-1]]
        for i,k in enumerate(keys2):
            res[values2[i]] = end_points[f"resnet_v2_{self.cfg.MODEL.RESNETS.DEPTH}/"+keys2[i]]

        next_nr = len(keys)+1
        if self.cfg.MODEL.RESNETS.MAKE_C6C7 == "C6":
            res[f"C{next_nr}"] = slim.avg_pool2d(res["C5"],kernel_size=1, stride=2, padding="SAME")
        elif self.cfg.MODEL.RESNETS.MAKE_C6C7 == "C6C7":
            with tf.variable_scope("FeatureExtractor"):
                last_feature = res[f"C{next_nr-1}"]
                for i in range(2):
                    last_feature = slim.conv2d(last_feature, self.out_channels, [3, 3], stride=2,
                                               activation_fn=self.activation_fn,
                                               normalizer_fn=self.normalizer_fn,
                                               normalizer_params=self.norm_params,
                                               scope=f"conv{i + 1}")
                    res[f"C{next_nr+i}"] = last_feature
        return res


@BACKBONE_REGISTRY.register()
def build_resnetv2_backbone(cfg, *args,**kwargs):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    return ResNetV2(cfg,*args,**kwargs)
