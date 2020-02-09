#coding=utf-8
from thirdparty.registry import Registry
PROPOSAL_GENERATOR_REGISTRY = Registry("PROPOSAL_GENERATOR")

def build_proposal_generator(cfg, *args,**kwargs):
    """
    Build a proposal generator from `cfg.MODEL.PROPOSAL_GENERATOR.NAME`.
    The name can be "PrecomputedProposals" to use no proposal generator.
    """
    name = cfg.MODEL.PROPOSAL_GENERATOR.NAME
    if name == "PrecomputedProposals":
        return None

    return PROPOSAL_GENERATOR_REGISTRY.get(name)(cfg, *args,**kwargs)