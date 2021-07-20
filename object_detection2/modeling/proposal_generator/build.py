#coding=utf-8
from thirdparty.registry import Registry
PROPOSAL_GENERATOR_REGISTRY = Registry("PROPOSAL_GENERATOR")
'''
Proposal的输入为backborn的输出，输出为outdata（key:PD_BOXES [B,N,4], key:PD_PROBABILITY可选[B,N]), loss
'''
def build_proposal_generator(cfg, *args,**kwargs):
    """
    Build a proposal generator from `cfg.MODEL.PROPOSAL_GENERATOR.NAME`.
    The name can be "PrecomputedProposals" to use no proposal generator.
    """
    name = cfg.MODEL.PROPOSAL_GENERATOR.NAME
    if name == "PrecomputedProposals":
        return None

    return PROPOSAL_GENERATOR_REGISTRY.get(name)(cfg, *args,**kwargs)

def build_proposal_generator_by_name(name,cfg, *args,**kwargs):
    """
    Build a proposal generator from `cfg.MODEL.PROPOSAL_GENERATOR.NAME`.
    The name can be "PrecomputedProposals" to use no proposal generator.
    """
    if name == "PrecomputedProposals":
        return None

    return PROPOSAL_GENERATOR_REGISTRY.get(name)(cfg, *args,**kwargs)
