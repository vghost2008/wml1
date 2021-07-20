#coding=utf-8
import tensorflow as tf
from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from object_detection2.od_toolkit import get_activation_fn
from thirdparty.nets.mobilenet import mobilenet_v3
import collections
import copy

slim = tf.contrib.slim


class MobileNet(Backbone):
    def __init__(self,cfg,*args,**kwargs):
        if cfg.MODEL.PREPROCESS != "ton1p1":
            print("--------------------WARNING--------------------")
            print(f"Preprocess for mobilenet should be ton1p1 not {cfg.MODEL.PREPROCESS}.")
            print("------------------END WARNING------------------")

        super().__init__(cfg,*args,**kwargs)

    def forward(self, x):
        res = collections.OrderedDict()
        batch_norm_decay = self.cfg.MODEL.MOBILENETS.batch_norm_decay #0.999
        activation_fn = get_activation_fn(self.cfg.MODEL.MOBILENETS.ACTIVATION_FN)
        if not self.is_training:
            train_bn = False
        elif self.cfg.MODEL.MOBILENETS.FROZEN_BN:
            print("Frozen bn.")
            train_bn = False
        else:
            train_bn = True
        MINOR_VERSION = self.cfg.MODEL.MOBILENETS.MINOR_VERSION
        with slim.arg_scope(mobilenet_v3.training_scope(bn_decay=batch_norm_decay,
                                                        is_training=train_bn)):
            if MINOR_VERSION=="LARGE":
                conv_defs=mobilenet_v3.V3_LARGE
                keys = ["layer_2", "layer_4", "layer_7", "layer_13", "layer_17"]
            elif MINOR_VERSION == "LARGE_DETECTION":
                conv_defs=mobilenet_v3.V3_LARGE_DETECTION
                keys = ["layer_2", "layer_4", "layer_7", "layer_13", "layer_17"]
            elif MINOR_VERSION=="SMALL":
                conv_defs=mobilenet_v3.get_V3_SMALL(activation_fn)
                keys = ["layer_1", "layer_2", "layer_4", "layer_9", "layer_13"]
            elif MINOR_VERSION=="SMALL_DETECTION":
                conv_defs=mobilenet_v3.get_V3_SMALL_DETECTION(activation_fn)
                keys = ["layer_1", "layer_2", "layer_4", "layer_9", "layer_13"]
            else:
                conv_defs = None
                print(f"ERROR MobileNet Minor version {MINOR_VERSION}")
        if activation_fn is not None:
            defs = copy.deepcopy(conv_defs)
            for d in defs['spec']:
                if 'activation_fn' in d.params:
                    d.params.update({
                        'activation_fn':activation_fn
                    })

        with slim.arg_scope(mobilenet_v3.training_scope(bn_decay=batch_norm_decay,
                                             is_training=train_bn)):
                _,end_points = mobilenet_v3.mobilenet(x['image'],output_stride=None,
                                                      base_only=True,
                                                      num_classes=None,
                                                      conv_defs=defs)

        self.end_points = end_points
        res.update(end_points)

        for i in range(1,6):
            res[f"C{i}"] = end_points[keys[i-1]]

        return res


@BACKBONE_REGISTRY.register()
def build_mobile_backbone(cfg, *args,**kwargs):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    return MobileNet(cfg,*args,**kwargs)
