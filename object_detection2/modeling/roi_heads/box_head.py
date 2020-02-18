import numpy as np
import tensorflow as tf
from thirdparty.registry import Registry
import wmodule
import wml_tfutils as wmlt
import wnnlayer as wnnl

slim = tf.contrib.slim

ROI_BOX_HEAD_REGISTRY = Registry("ROI_BOX_HEAD")
ROI_BOX_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHead(wmodule.WChildModule):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg,*args,**kwargs):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__(cfg,*args,**kwargs)
        '''
        Detectron2是没有使用任何normalizer的
        但我在使用测试数据时发现，使用group_norm(默认参数)后性能会显著提升
        '''
        self.normalizer_fn = wnnl.group_norm


    def forward(self, x,scope="FastRCNNConvFCHead"):
        with tf.variable_scope(scope):
            cfg = self.cfg
            conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
            num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
            fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
            num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC

            assert num_conv + num_fc > 0

            for _ in range(num_conv):
                x = slim.conv2d(x,conv_dim,[3,3],activation_fn=tf.nn.relu,
                                normalizer_fn=None)
            if num_fc>0:
                if len(x.get_shape()) > 2:
                    shape = wmlt.combined_static_and_dynamic_shape(x)
                    x = tf.reshape(x,[shape[0],-1])
                for _ in range(num_fc):
                    x = slim.fully_connected(x,fc_dim,activation_fn=tf.nn.relu,
                                             normalizer_fn=self.normalizer_fn)

            return x

    @property
    def output_size(self):
        return self._output_size


def build_box_head(cfg, *args,**kwargs):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, *args,**kwargs)
