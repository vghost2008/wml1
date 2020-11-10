#coding=utf-8
import tensorflow as tf
import wmodule
from collections import OrderedDict
from .build import ROI_HEADS_HOOK
import wnnlayer as wnnl

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
        net = wnnl.non_local_blockv1(net,scope=f"NonLocalROIHeadsHook",
                                         inner_dims_multiplier=[1,1,1],
                                         normalizer_fn=wnnl.evo_norm_s0,
                                         activation_fn=None,
                                         weighed_sum=False,
                                         n_head=8)
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
class OneHeadSEROIHeadsHook(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent,*args,**kwargs)

    def forward(self,net,batched_inputs,reuse=None):
        del batched_inputs
        net = wnnl.se_block(net,scope=f"SEROIHeadsHook")
        return net
