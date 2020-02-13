#coding=utf-8
import tensorflow as tf
from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from thirdparty.nets.shufflenetv2 import shufflenetv2
import collections

slim = tf.contrib.slim


class ShuffleNetV2(Backbone):
    def __init__(self,cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)

    def forward(self, x):
        res = collections.OrderedDict()
        _,end_points = shufflenetv2(x['image'],is_training=self.is_training,
                                    num_classes=None)

        self.end_points = end_points

        for i in range(1,5):
            res[f"C{i}"] = end_points["ShuffleNetV2/"+f"Stage{i}"]

        return res


@BACKBONE_REGISTRY.register()
def build_shufflenetv2_backbone(cfg, *args,**kwargs):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    return ShuffleNetV2(cfg,*args,**kwargs)
