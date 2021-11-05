from thirdparty.registry import Registry

MOT_REGISTRY = Registry("MOT")

def build_mot(cfg, model,*args,**kwargs):
    name = cfg.MODEL.MOT.NAME
    mot = MOT_REGISTRY.get(name)(model, *args, **kwargs)
    return mot
