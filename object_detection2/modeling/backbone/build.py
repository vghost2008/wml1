from thirdparty.registry import Registry
from .backbone import Backbone

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_HOOK_REGISTRY = Registry("BACKBONE_HOOK")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images

The registered object must be a callable that accepts two arguments:

1. A :class:`config.CfgNode`
2. A :class:`layers.ShapeSpec`, which contains the input shape specification.

It must returns an instance of :class:`Backbone`.
"""


def build_backbone(cfg, *args,**kwargs):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg,*args,**kwargs)
    assert isinstance(backbone, Backbone)
    return backbone

def build_backbone_hook(cfg, *args,**kwargs):
    """
    cfg: only child part
    Returns:
        an instance of :class:`Backbone`
    """
    bhn0,bhn1 = cfg.BACKBONE_HOOK
    hook0 = None
    hook1 = None
    if len(bhn0) > 0:
        hook0 = BACKBONE_HOOK_REGISTRY.get(bhn0)(cfg,*args,**kwargs)
    if len(bhn1) > 0:
        hook1 = BACKBONE_HOOK_REGISTRY.get(bhn1)(cfg,*args,**kwargs)
    return hook0,hook1
