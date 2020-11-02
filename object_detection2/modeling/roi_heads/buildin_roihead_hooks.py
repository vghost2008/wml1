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
