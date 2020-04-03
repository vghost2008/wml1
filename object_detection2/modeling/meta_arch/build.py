from thirdparty.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
HEAD_OUTPUTS = Registry("HEAD_OUTPUTS")


def build_model(cfg,**kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    return META_ARCH_REGISTRY.get(meta_arch)(cfg,**kwargs)

def build_outputs(name,*args,**kwargs):
    outputs = HEAD_OUTPUTS.get(name)(*args,**kwargs)
    return outputs