#coding=utf-8
import tensorflow as tf
from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from thirdparty.nets.shufflenetv2 import shufflenetv2
import collections
import object_detection2.od_toolkit as odt

slim = tf.contrib.slim


class ShuffleNetV2(Backbone):
    def __init__(self,cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)
        self.out_channels = self.cfg.MODEL.SHUFFLENETS.OUT_CHANNELS
        self.normalizer_fn, self.norm_params = odt.get_norm(self.cfg.MODEL.SHUFFLENETS.NORM, self.is_training)
        self.activation_fn = odt.get_activation_fn(self.cfg.MODEL.SHUFFLENETS.ACTIVATION_FN)



    def forward(self, x):
        res = collections.OrderedDict()
        _,end_points = shufflenetv2(x['image'],is_training=self.is_training,
                                    use_max_pool=self.cfg.MODEL.SHUFFLENETS.use_max_pool,
                                    later_max_pool=self.cfg.MODEL.SHUFFLENETS.later_max_pool,
                                    num_classes=None)

        self.end_points = end_points
        stage = [1,2,3,4]
        if self.cfg.MODEL.SHUFFLENETS.use_max_pool:
            for i in range(1,4):
                stage[i] += 1
        if self.cfg.MODEL.SHUFFLENETS.later_max_pool:
            for i in range(2,4):
                stage[i] += 1

        for i in range(1,5):
            res[f"C{stage[i-1]}"] = end_points["ShuffleNetV2/"+f"Stage{i}"]

        if self.cfg.MODEL.SHUFFLENETS.MAKE_C6C7 == "C6":
            res[f"C{stage[-1]+1}"] = slim.avg_pool2d(res.keys()[-1], kernel_size=1, stride=2, padding="SAME")
        elif self.cfg.MODEL.SHUFFLENETS.MAKE_C6C7 == "C6C7":
            with tf.variable_scope("FeatureExtractor"):
                last_feature = list(res.values())[-1]
                for i in range(2):
                    last_feature = slim.conv2d(last_feature, self.out_channels, [3, 3], stride=2,
                                               activation_fn=self.activation_fn,
                                               normalizer_fn=self.normalizer_fn,
                                               normalizer_params=self.norm_params,
                                               scope=f"conv{i + 1}")
                    res[f"C{stage[-1] + 1 + i}"] = last_feature

        return res


@BACKBONE_REGISTRY.register()
def build_shufflenetv2_backbone(cfg, *args,**kwargs):
    """
    Create a ShuffleNetV2 instance from config.
    """
    # need registration of new blocks/stems?
    return ShuffleNetV2(cfg,*args,**kwargs)
