#coding=utf-8
import tensorflow as tf
from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from thirdparty.nets.resnet_v1 import *
import collections
import wmodule
import object_detection2.od_toolkit as odt
from wnets.hrnet import HighResolutionNet,mini_cfg_w32

slim = tf.contrib.slim


class HRNet(Backbone):
    def __init__(self,cfg,is_mini=False,**kwargs):
        super().__init__(cfg,**kwargs)
        self.is_mini = is_mini

    def forward(self, x):
        net = x['image']
        net = tf.image.resize_bilinear(net,(512,512))
        if self.is_mini:
            end_points = HighResolutionNet(mini_cfg_w32,output_channel=256)(net)
        else:
            end_points = HighResolutionNet(output_channel=256)(net)
        return end_points


@BACKBONE_REGISTRY.register()
def build_hrnet_backbone(cfg, *args,**kwargs):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    return HRNet(cfg,*args,**kwargs)

@BACKBONE_REGISTRY.register()
def build_mini_hrnet_backbone(cfg, *args,**kwargs):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    return HRNet(cfg,*args,is_mini=True,**kwargs)
