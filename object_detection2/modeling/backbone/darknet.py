#coding=utf-8
import tensorflow as tf
from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from wnets.darknets import CSPDarkNet
import collections
import wmodule
import object_detection2.od_toolkit as odt

slim = tf.contrib.slim


class DarkNet(Backbone):
    def __init__(self,cfg,*args,**kwargs):
        if cfg.MODEL.PREPROCESS != "ton1p1":
            print("--------------------WARNING--------------------")
            print(f"Preprocess for mobilenet should be ton1p1 not {cfg.MODEL.PREPROCESS}.")
            print("------------------END WARNING------------------")
        super().__init__(cfg,*args,**kwargs)
        self.normalizer_fn, self.norm_params = odt.get_norm(self.cfg.MODEL.DARKNETS.NORM, self.is_training)
        self.activation_fn = odt.get_activation_fn(self.cfg.MODEL.DARKNETS.ACTIVATION_FN)
        self.out_channels = cfg.MODEL.DARKNETS.OUT_CHANNELS
        self.scope_name = "50"

    def forward(self, x):
        res = collections.OrderedDict()
        if self.cfg.MODEL.DARKNETS.DEPTH == 53:
            print("DarkNet-53")
            darknet = CSPDarkNet(normalizer_fn=self.normalizer_fn,
                                 normalizer_params=self.norm_params,
                                 activation_fn=self.activation_fn)
            _,end_points = darknet.forward(x['image'],scope=f"CSPDarkNet-53")
        else:
            print(f"Error Depth {self.cfg.MODEL.DARKNETS.DEPTH}")
            return None

        self.end_points = end_points
        res.update(end_points)

        level = int(list(end_points.keys())[-1][1:]) + 1
        x = list(self.end_points.values())[-1]
        for i in range(self.cfg.MODEL.DARKNETS.ADD_CONV):
            res[f"C{level+i}"] = slim.conv2d(x, self.out_channels, [3, 3], stride=2,
                                       activation_fn=self.activation_fn,
                                       normalizer_fn=self.normalizer_fn,
                                       normalizer_params=self.norm_params,
                                       scope=f"conv{i}")
        return res


@BACKBONE_REGISTRY.register()
def build_darknet_backbone(cfg, *args,**kwargs):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    return DarkNet(cfg,*args,**kwargs)
