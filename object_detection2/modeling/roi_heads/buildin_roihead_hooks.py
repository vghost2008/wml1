#coding=utf-8
import tensorflow as tf
import wmodule
from collections import OrderedDict
from .build import ROI_HEADS_HOOK
import wnnlayer as wnnl
import object_detection2.od_toolkit as odt
import basic_tftools as btf

slim = tf.contrib.slim

@ROI_HEADS_HOOK.register()
class NonLocalROIHeadsHook(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,net,batched_inputs):
        del batched_inputs
        cls_net = wnnl.non_local_blockv1(net,scope=f"NonLocalROIHeadsHook_cls",
                                         normalizer_fn=wnnl.evo_norm_s0,
                                         activation_fn=None,
                                         weighed_sum=False)
        reg_net = wnnl.non_local_blockv1(net,scope=f"NonLocalROIHeadsHook_reg",
                                         normalizer_fn=wnnl.evo_norm_s0,
                                         activation_fn=None,
                                         weighed_sum=False)
        return cls_net,reg_net

@ROI_HEADS_HOOK.register()
class NonLocalROIHeadsHookV2(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,net,batched_inputs):
        del batched_inputs
        cls_net = wnnl.non_local_blockv4(net,scope=f"NonLocalROIHeadsHook_clsv2",
                                         normalizer_fn=wnnl.evo_norm_s0,
                                         activation_fn=None,
                                         n_head=4,
                                         weighed_sum=False)
        reg_net = wnnl.non_local_blockv4(net,scope=f"NonLocalROIHeadsHook_regv2",
                                         normalizer_fn=wnnl.evo_norm_s0,
                                         n_head=4,
                                         activation_fn=None,
                                         weighed_sum=False)
        return cls_net,reg_net

@ROI_HEADS_HOOK.register()
class OneHeadNonLocalROIHeadsHook(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,net,batched_inputs,reuse=None):
        del batched_inputs
        net = wnnl.non_local_blockv1(net,scope=f"NonLocalROIHeadsHook",
                                         normalizer_fn=wnnl.evo_norm_s0,
                                         activation_fn=None,
                                         weighed_sum=False)
        return net

@ROI_HEADS_HOOK.register()
class OneHeadNonLocalROIHeadsHookV2(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,net,batched_inputs,reuse=None):
        del batched_inputs
        net = wnnl.non_local_blockv4(net,scope=f"NonLocalROIHeadsHookV2",
                                         inner_dims_multiplier=[1,1,1],
                                         normalizer_fn=wnnl.evo_norm_s0,
                                         activation_fn=None,
                                         weighed_sum=False,
                                         n_head=4)
        return net

@ROI_HEADS_HOOK.register()
class OneHeadNonLocalROIHeadsHookV3(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,net,batched_inputs,reuse=None):
        del batched_inputs
        net = wnnl.non_local_blockv1(net,scope=f"NonLocalROIHeadsHook",
                                         inner_dims_multiplier=[1,1,1],
                                         normalizer_fn=wnnl.evo_norm_s0,
                                         activation_fn=None,
                                         weighed_sum=False)
        return net

@ROI_HEADS_HOOK.register()
class ClsNonLocalROIHeadsHook(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,net,batched_inputs):
        del batched_inputs
        cls_net = wnnl.non_local_blockv1(net,scope=f"NonLocalROIHeadsHook_cls",
                                         normalizer_fn=wnnl.evo_norm_s0,
                                         activation_fn=None,
                                         weighed_sum=False)
        return cls_net,net

@ROI_HEADS_HOOK.register()
class ClsNonLocalROIHeadsHookV2(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,net,batched_inputs):
        del batched_inputs
        cls_net = wnnl.non_local_blockv4(net,scope=f"NonLocalROIHeadsHook_clsv2",
                                         normalizer_fn=wnnl.evo_norm_s0,
                                         activation_fn=None,
                                         n_head=4,
                                         weighed_sum=False)
        return cls_net,net

@ROI_HEADS_HOOK.register()
class ClsNonLocalROIHeadsHookV3(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,net,batched_inputs):
        del batched_inputs
        cls_net = wnnl.non_local_blockv4(net,scope=f"NonLocalROIHeadsHook_clsv3",
                                         inner_dims_multiplier=[2,2,2],
                                         normalizer_fn=wnnl.evo_norm_s0,
                                         activation_fn=None,
                                         n_head=2,
                                         weighed_sum=False)
        return cls_net,net

@ROI_HEADS_HOOK.register()
class BoxNonLocalROIHeadsHook(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,net,batched_inputs):
        del batched_inputs
        reg_net = wnnl.non_local_blockv1(net,scope=f"NonLocalROIHeadsHook_reg",
                                         normalizer_fn=wnnl.evo_norm_s0,
                                         activation_fn=None,
                                         weighed_sum=False)
        return net,reg_net

@ROI_HEADS_HOOK.register()
class BoxNonLocalROIHeadsHookV2(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,net,batched_inputs):
        del batched_inputs
        reg_net = wnnl.non_local_blockv4(net,scope=f"NonLocalROIHeadsHook_regv2",
                                         normalizer_fn=wnnl.evo_norm_s0,
                                         activation_fn=None,
                                         n_head=4,
                                         weighed_sum=False)
        return net,reg_net
    
@ROI_HEADS_HOOK.register()
class IouHeadNonLocalROIHeadsHook(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,x,batched_inputs,reuse=None):
        del batched_inputs
        if isinstance(x,(list,tuple)) and len(x) == 2:
            iou_x = wnnl.non_local_blockv3(x[0], x[1], x[1], inner_dims_multiplier=[1, 1, 1],
                                           scope=f"NonLocalROIHeadsHook_iou",
                                           normalizer_fn=wnnl.evo_norm_s0,
                                           activation_fn=None,
                                           weighed_sum=False,
                                           skip_connect=False)
            return x[0],x[0],iou_x
        else:
            iou_x = wnnl.non_local_blockv1(x, scope=f"NonLocalROIHeadsHook_iou",
                                             normalizer_fn=wnnl.evo_norm_s0,
                                             activation_fn=None,
                                             weighed_sum=False)
            return x,x,iou_x
        
@ROI_HEADS_HOOK.register()
class SEROIHeadsHook(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,net,batched_inputs):
        del batched_inputs
        cls_net = wnnl.se_block(net,scope=f"SEROIHeadsHook_cls")
        reg_net = wnnl.se_block(net,scope=f"SEROIHeadsHook_reg")
        return cls_net,reg_net

@ROI_HEADS_HOOK.register()
class ClsSEROIHeadsHook(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,net,batched_inputs):
        del batched_inputs
        cls_net = wnnl.se_block(net,scope=f"SEROIHeadsHook_cls")
        reg_net = net
        return cls_net,reg_net
    
@ROI_HEADS_HOOK.register()
class OneHeadSEROIHeadsHook(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,net,batched_inputs,reuse=None):
        del batched_inputs
        net = wnnl.se_block(net,scope=f"SEROIHeadsHook")
        return net
    
@ROI_HEADS_HOOK.register()
class OneHeadCBAMROIHeadsHook(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,net,batched_inputs,reuse=None):
        del batched_inputs
        net = wnnl.cbam_block(net,scope=f"CBAMROIHeadsHook")
        return net

@ROI_HEADS_HOOK.register()
class AddBBoxesSizeInfo(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)
        self.normalizer_fn, self.norm_params = odt.get_norm(self.cfg.MODEL.ROI_BOX_HEAD.NORM, self.is_training)
        self.activation_fn = odt.get_activation_fn(self.cfg.MODEL.ROI_BOX_HEAD.ACTIVATION_FN)

    def forward(self,net,batched_inputs,reuse=None):
        with tf.variable_scope("AddBBoxesSizeInfo"):
            C = btf.channel(net)
            bboxes = self.parent.t_proposal_boxes
            B,BN,BC = btf.combined_static_and_dynamic_shape(bboxes)
            bboxes = tf.reshape(bboxes,[B*BN,BC])
            bboxes = slim.fully_connected(bboxes,C*2,activation_fn=self.activation_fn,
                                          normalizer_fn=self.normalizer_fn,
                                          normalizer_params=self.norm_params)
            bboxes = slim.fully_connected(bboxes,C*2,activation_fn=None,
                                          normalizer_fn=None)
            gamma = bboxes[...,:C]
            beta = bboxes[...,C:]
            net = wnnl.group_norm_v2(net,gamma,beta)
            return net

