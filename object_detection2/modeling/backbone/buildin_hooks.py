#coding=utf-8
import tensorflow as tf
import wnnlayer as wnnl
import wmodule
import wml_tfutils as wmlt
from collections import OrderedDict
import object_detection2.od_toolkit as odt
from object_detection2.config.config import global_cfg
import basic_tftools as btf
from object_detection2.modeling.backbone.build import BACKBONE_HOOK_REGISTRY

slim = tf.contrib.slim

@BACKBONE_HOOK_REGISTRY.register()
class NonLocalBackboneHook(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,features,batched_inputs):
        del batched_inputs
        res = OrderedDict()
        normalizer_fn, normalizer_params = odt.get_norm("evo_norm_s0", is_training=self.is_training)
        with tf.variable_scope("NonLocalBackboneHook"):
            for k,v in features.items():
                if k[0] not in ["C","P"]:
                    continue
                level = int(k[1:])
                if level<=3:
                    res[k] = v
                    continue
                res[k]= wnnl.non_local_blockv1(v, inner_dims_multiplier=[1, 1, 1],
                                         normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                                         activation_fn=None,
                                         weighed_sum=False)
            return res

@BACKBONE_HOOK_REGISTRY.register()
class NonLocalBackboneHookV2(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,features,batched_inputs):
        del batched_inputs
        res = OrderedDict()
        normalizer_fn, normalizer_params = odt.get_norm("evo_norm_s0", is_training=self.is_training)
        with tf.variable_scope("NonLocalBackboneHookV2"):
            for k,v in features.items():
                if k[0] not in ["C","P"]:
                    continue
                level = int(k[1:])
                if level<=3:
                    res[k] = v
                    continue
                res[k]= wnnl.non_local_blockv1(v,
                                               inner_dims=[128, 128, 128],
                                               normalizer_fn=normalizer_fn,
                                               normalizer_params=normalizer_params,
                                               n_head=2,
                                               activation_fn=None,
                                               weighed_sum=False)
            return res

@BACKBONE_HOOK_REGISTRY.register()
class NonLocalBackboneHookV3(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)
        self.base_size = 512

    def forward(self,features,batched_inputs):
        del batched_inputs
        res = OrderedDict()
        normalizer_fn, normalizer_params = odt.get_norm("evo_norm_s0", is_training=self.is_training)
        normalizer_params['G'] = 8
        with tf.variable_scope("NonLocalBackboneHookV3"):
            for k,v in features.items():
                if k[0] not in ["C","P"]:
                    continue
                level = int(k[1:])
                if level<=2:
                    res[k] = v
                    continue
                h = self.base_size//(2**level)
                w = self.base_size//(2**level)
                v = wnnl.non_local_blockv4(v,
                                           inner_dims=[128, 128, 128],
                                           normalizer_fn=normalizer_fn,
                                           normalizer_params=normalizer_params,
                                           n_head=2,
                                           activation_fn=None,
                                           weighed_sum=False,
                                           scope=f"non_localv4_{level}",
                                           size=[h,w])
                res[k] = v
            return res

@BACKBONE_HOOK_REGISTRY.register()
class SEBackboneHook(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,features,batched_inputs):
        del batched_inputs
        res = OrderedDict()
        with tf.variable_scope("SEBackboneHook"):
            for k,v in features.items():
                if k[0] not in ["C","P"]:
                    continue
                level = int(k[1:])
                if level<=3:
                    res[k] = v
                    continue
                res[k]= wnnl.se_block(v,scope=f"SE_block_{k}")
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
class FusionBackboneHookV2(wmodule.WChildModule):
    def __init__(self, cfg, parent, *args, **kwargs):
        super().__init__(cfg, parent, *args, **kwargs)

    def forward(self, features, batched_inputs):
        normalizer_fn,normalizer_params = odt.get_norm("evo_norm_s0",is_training=self.is_training)
        with tf.variable_scope("FusionBackboneHookV2"):
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
            net = slim.conv2d(net, v0.get_shape().as_list()[-1], [3, 3],
                              activation_fn=None,
                              normalizer_fn=normalizer_fn,
                              normalizer_params=normalizer_params,
                              scope=f"smooth{level0}")
            res = features
            res[f'F{level0}'] = net

            return res

@BACKBONE_HOOK_REGISTRY.register()
class BalanceBackboneHook(wmodule.WChildModule):
    def __init__(self, cfg, parent, *args, **kwargs):
        super().__init__(cfg, parent, *args, **kwargs)

    def forward(self, features, batched_inputs):
        normalizer_fn,normalizer_params = odt.get_norm("evo_norm_s0",is_training=self.is_training)
        res = OrderedDict()
        with tf.variable_scope("BalanceBackboneHook"):
            del batched_inputs
            ref_index = 1
            end_points = list(features.items())
            k0,v0 = end_points[ref_index]
            mfeatures = []
            with tf.name_scope("fusion"):
                shape0 = wmlt.combined_static_and_dynamic_shape(v0)
                for i,(k, v) in enumerate(end_points):
                    if i == ref_index:
                        net = v
                    else:
                        net = tf.image.resize_bilinear(v,shape0[1:3],name=f"resize{i}")
                    mfeatures.append(net)
                net = tf.add_n(mfeatures)/float(len(mfeatures))
                net = slim.conv2d(net, net.get_shape().as_list()[-1], [3, 3],
                                  activation_fn=None,
                                  normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_params,
                                  scope=f"smooth")
            for i,(k,v) in enumerate(end_points):
                with tf.name_scope(f"merge{i}"):
                    shape = wmlt.combined_static_and_dynamic_shape(v)
                    v0 = tf.image.resize_bilinear(net,shape[1:3])
                    res[k] = v+v0
                
            return res


@BACKBONE_HOOK_REGISTRY.register()
class BalanceBackboneHookV2(wmodule.WChildModule):
    def __init__(self, cfg, parent, *args, **kwargs):
        super().__init__(cfg, parent, *args, **kwargs)

    def forward(self, features, batched_inputs):
        low_features = self.parent.low_features
        normalizer_fn, normalizer_params = odt.get_norm("evo_norm_s0", is_training=self.is_training)
        res = OrderedDict()
        with tf.variable_scope("BalanceBackboneHookV2"):
            del batched_inputs
            ref_index = 1
            end_points = list(features.items())
            k0, v0 = end_points[ref_index]
            mfeatures = []
            with tf.name_scope("fusion"):
                shape0 = wmlt.combined_static_and_dynamic_shape(v0)
                for i, (k, v) in enumerate(end_points):
                    if i == ref_index:
                        net = v
                    else:
                        net = tf.image.resize_bilinear(v, shape0[1:3], name=f"resize{i}")
                    mfeatures.append(net)
                net = tf.add_n(mfeatures) / float(len(mfeatures))
                net = slim.conv2d(net, net.get_shape().as_list()[-1], [3, 3],
                                  activation_fn=None,
                                  normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_params,
                                  scope=f"smooth")
            for i, (k, v) in enumerate(end_points):
                with tf.name_scope(f"smooth_low_feature{i}"):
                    index = int(k[1:])
                    low_feature = low_features[f"C{index}"]
                    channel = v.get_shape().as_list()[-1]
                    low_feature = slim.conv2d(low_feature,channel,[1,1],activation_fn=None,
                                              normalizer_fn=None)
                with tf.name_scope(f"merge{i}"):
                    shape = wmlt.combined_static_and_dynamic_shape(v)
                    v0 = tf.image.resize_bilinear(net, shape[1:3])
                    res[k] = tf.concat([v + v0,low_feature],axis=-1)

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

@BACKBONE_HOOK_REGISTRY.register()
class MakeAnchorsForRetinaNet(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)
        self.bh = BalanceBackboneHook(cfg,parent,*args,**kwargs)

    def forward(self,features,batched_inputs):
        features = self.bh(features,batched_inputs)
        del batched_inputs
        res = OrderedDict()
        featuremap_keys =  ["P3","P4","P5","P6","P7"]
        anchor_sizes = global_cfg.MODEL.ANCHOR_GENERATOR.SIZES
        anchor_ratios = global_cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS
        normalizer_fn,normalizer_params = odt.get_norm("evo_norm_s0",is_training=self.is_training)
        ref = features[featuremap_keys[1]]
        ref_shape = wmlt.combined_static_and_dynamic_shape(ref)[1:3]
        ref_size = anchor_sizes[1][0]
        nr = 0
        with tf.name_scope("MakeAnchorsForRetinaNet"):
            for i,k in enumerate(featuremap_keys):
                net = features[k]
                for j,s in enumerate(anchor_sizes[i]):
                    for k,r in enumerate(anchor_ratios[i][j]):
                        net = slim.separable_conv2d(net, 32,
                                            kernel_size=3,
                                            padding="SAME",
                                            depth_multiplier=1,
                                            normalizer_fn=normalizer_fn,
                                            normalizer_params=normalizer_params,
                                            scope=f"sep_conv_{i}{j}{k}")
                        target_shape = self.get_shape(ref_shape,ref_size,s,r)
                        net = tf.image.resize_nearest_neighbor(net,target_shape)
                        res[f"P{nr}"] = net
                        nr += 1
        return res

    @staticmethod
    @wmlt.add_name_scope
    def get_shape(ref_shape,ref_size,size,ratio):
        ref_size = tf.to_float(ref_size)
        size = tf.to_float(size)
        ref_shape = tf.to_float(ref_shape)
        target_shape = (ref_size/size)*ref_shape*tf.stack([tf.sqrt(ratio),tf.rsqrt(ratio)],axis=0)
        shape = tf.to_int32(target_shape)
        return tf.where(shape>0,shape,tf.ones_like(shape))


@BACKBONE_HOOK_REGISTRY.register()
class BalanceBackboneHookV3(wmodule.WChildModule):
    def __init__(self, cfg, parent, *args, **kwargs):
        super().__init__(cfg, parent, *args, **kwargs)

    def forward(self, features, batched_inputs):
        normalizer_fn, normalizer_params = odt.get_norm("evo_norm_s0", is_training=self.is_training)
        res = []
        with tf.variable_scope("BalanceBackboneHook"):
            del batched_inputs
            ref_index = 1
            end_points = list(features)
            v0 = end_points[ref_index]
            mfeatures = []
            with tf.name_scope("fusion"):
                shape0 = wmlt.combined_static_and_dynamic_shape(v0)
                for i, v in enumerate(end_points):
                    if i == ref_index:
                        net = v
                    else:
                        net = tf.image.resize_bilinear(v, shape0[1:3], name=f"resize{i}")
                    mfeatures.append(net)
                net = tf.add_n(mfeatures) / float(len(mfeatures))
                net = slim.conv2d(net, net.get_shape().as_list()[-1], [3, 3],
                                  activation_fn=None,
                                  normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_params,
                                  scope=f"smooth")
            for i, v in enumerate(end_points):
                with tf.name_scope(f"merge{i}"):
                    shape = wmlt.combined_static_and_dynamic_shape(v)
                    v0 = tf.image.resize_bilinear(net, shape[1:3])
                    res.append(v + v0)

            return res


@BACKBONE_HOOK_REGISTRY.register()
class BalanceNonLocalBackboneHook(wmodule.WChildModule):
    def __init__(self, cfg, parent, *args, **kwargs):
        super().__init__(cfg, parent, *args, **kwargs)

    def forward(self, features, batched_inputs):
        normalizer_fn, normalizer_params = odt.get_norm("evo_norm_s0", is_training=self.is_training)
        res = OrderedDict()
        with tf.variable_scope("BalanceNonLocalBackboneHook"):
            del batched_inputs
            ref_index = 1
            end_points = list(features.items())
            k0, v0 = end_points[ref_index]
            mfeatures = []
            with tf.name_scope("fusion"):
                shape0 = wmlt.combined_static_and_dynamic_shape(v0)
                for i, (k, v) in enumerate(end_points):
                    if i == ref_index:
                        net = v
                    else:
                        net = tf.image.resize_bilinear(v, shape0[1:3], name=f"resize{i}")
                    mfeatures.append(net)
                net = tf.add_n(mfeatures) / float(len(mfeatures))
                net = slim.conv2d(net, net.get_shape().as_list()[-1], [3, 3],
                                  activation_fn=None,
                                  normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_params,
                                  scope=f"smooth")
            for i, (k, v) in enumerate(end_points):
                with tf.variable_scope(f"merge{i}"):
                    shape = wmlt.combined_static_and_dynamic_shape(v)
                    v0 = tf.image.resize_bilinear(net, shape[1:3])
                    net = v + v0
                    if i>0:
                        net = wnnl.non_local_blockv1(net,inner_dims_multiplier=[1,1,1],
                                                 normalizer_fn=normalizer_fn,normalizer_params=normalizer_params,
                                                 activation_fn=None,
                                                 weighed_sum=False)
                    res[k] = net

            return res
