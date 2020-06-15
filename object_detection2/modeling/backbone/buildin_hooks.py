#coding=utf-8
import tensorflow as tf
import wnnlayer as wnnl
import wmodule
import wml_tfutils as wmlt
from collections import OrderedDict
import object_detection2.od_toolkit as odt
slim = tf.contrib.slim
from object_detection2.modeling.backbone.build import BACKBONE_HOOK_REGISTRY

@BACKBONE_HOOK_REGISTRY.register()
class NonLocalBackboneHook(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,features,batched_inputs):
        del batched_inputs
        res = OrderedDict()
        with tf.variable_scope("NonLocalBackboneHook"):
            for k,v in features.items():
                if k[0] not in ["C","P"]:
                    continue
                level = int(k[1:])
                if level<=3:
                    res[k] = v
                    continue
                res[k] = wnnl.non_local_blockv1(v,scope=f"Non_Local{level}",
                                                    normalizer_fn=wnnl.group_norm)
            return res


@BACKBONE_HOOK_REGISTRY.register()
class FusionBackboneHook(wmodule.WChildModule):
    def __init__(self, cfg, parent, *args, **kwargs):
        super().__init__(cfg, parent, *args, **kwargs)

    def forward(self, features, batched_inputs):
        normalizer_fn,normalizer_params = odt.get_norm("evo_norm_s0",is_training=self.is_training)
        with tf.variable_scope("FusionBackboneHook"):
            del batched_inputs
            end_points = list(features.items())
            k0,v0 = end_points[0]
            mfeatures = []
            shape0 = wmlt.combined_static_and_dynamic_shape(v0)
            for k, v in end_points[1:]:
                net = tf.image.resize_bilinear(v,shape0[1:3])
                mfeatures.append(net)
            net = tf.add_n(mfeatures)/float(len(mfeatures))
            net = tf.concat([v0,net],axis=-1)
            level0 = int(k0[1:])
            net = slim.conv2d(net, net.get_shape().as_list()[-1], [3, 3],
                              activation_fn=None,
                              normalizer_fn=normalizer_fn,
                              normalizer_params=normalizer_params,
                              scope=f"smooth{level0}")
            res = features
            res[f'F{level0}'] = net

            return res
        
@BACKBONE_HOOK_REGISTRY.register()
class DeformConvBackboneHook(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,features,batched_inputs):
        del batched_inputs
        res = OrderedDict()
        with tf.variable_scope("DeformConvBackboneHook"):
            normalizer_fn,normalizer_params = odt.get_norm("BN",is_training=self.is_training)
            for k,v in features.items():
                if k[0] not in ["C", "P"]:
                    continue
                level = int(k[1:])
                channel = v.get_shape().as_list()[-1]
                res[k] = wnnl.deform_conv2dv2(v,num_outputs=channel,kernel_size=3,
                                              scope=f"deform_conv2d{level}",
                                                  normalizer_fn=normalizer_fn,
                                                  normalizer_params=normalizer_params)
            return res
