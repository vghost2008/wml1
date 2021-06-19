from thirdparty.registry import Registry

MOT_REGISTRY = Registry("MOT")

def build_mot(cfg, *args,**kwargs):
    name = cfg.MODEL.BACKBONE.NAME
    mot = MOT_REGISTRY.get(name)(cfg, *args, **kwargs)
    return mot
