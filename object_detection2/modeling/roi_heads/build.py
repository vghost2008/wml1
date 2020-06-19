from thirdparty.registry import Registry

ROI_HEADS_HOOK = Registry("ROI_HEADS_HOOK")

def build_roi_heads_hook(cfg, *args,**kwargs):
    name = cfg.MODEL.ROI_HEADS.HOOK
    if len(name) > 0:
        return ROI_HEADS_HOOK.get(name)(cfg,*args,**kwargs)
    else:
        return None

