#coding=utf-8
import tensorflow as tf
from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from thirdparty.nets.vgg import *
import collections
import wmodule
import object_detection2.od_toolkit as odt

slim = tf.contrib.slim


class VGG(Backbone):
    def __init__(self,type,cfg,*args,**kwargs):
        if cfg.MODEL.PREPROCESS != "subimagenetmean":
            print("--------------------WARNING--------------------")
            print(f"Preprocess for resnet should be subimagenetmean not {cfg.MODEL.PREPROCESS}.")
            print("------------------END WARNING------------------")
        super().__init__(cfg,*args,**kwargs)
        self.scope_name = "50"
        self.type = type

    def forward(self, x):
        res = collections.OrderedDict()
        with slim.arg_scope(vgg_arg_scope()):
                if self.type == 11:
                    print("VGG11")
                    _,end_points = vgg_a(x['image'],None)
                    ext_sc_name = "a"
                    keys = [
                        'conv2/conv2_1',
                        'conv3/conv3_2',
                        'conv4/conv4_2',
                        'conv5/conv5_2',
                    ]
                elif self.type == 16:
                    print("VGG16")
                    _, end_points = vgg_16(x['image'], None)
                    ext_sc_name = "16"
                    keys = [
                        'conv2/conv2_2',
                        'conv3/conv3_3',
                        'conv4/conv4_3',
                        'conv5/conv5_3',
                    ]
                elif self.type == 19:
                    print("VGG19")
                    _, end_points = vgg_19(x['image'], None)
                    ext_sc_name = "19"
                    keys = [
                        'conv2/conv2_2',
                        'conv3/conv3_4',
                        'conv4/conv4_4',
                        'conv5/conv5_4',
                    ]
                else:
                    print(f"ERROR vgg type {self.type}")

        self.end_points = end_points
        for i in range(1,len(keys)+1):
            res[f"C{i}"] = end_points[f"vgg_{ext_sc_name}/"+keys[i-1]]
        res.update(end_points)
        return res

@BACKBONE_REGISTRY.register()
def build_vgg11_backbone(cfg, *args,**kwargs):
    return VGG(11,cfg,*args,**kwargs)
@BACKBONE_REGISTRY.register()
def build_vgg16_backbone(cfg, *args,**kwargs):
    return VGG(16,cfg,*args,**kwargs)
@BACKBONE_REGISTRY.register()
def build_vgg19_backbone(cfg, *args,**kwargs):
    return VGG(19,cfg,*args,**kwargs)
