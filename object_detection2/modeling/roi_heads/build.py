from thirdparty.registry import Registry
from object_detection2.modeling.build import HEAD_OUTPUTS as _HEAD_OUTPUTS
from object_detection2.modeling.build import build_outputs as _build_outputs
from wmodule import WModelList

HEAD_OUTPUTS = _HEAD_OUTPUTS
ROI_HEADS_HOOK = Registry("ROI_HEADS_HOOK")
ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""
ROI_BOX_HEAD_REGISTRY = Registry("ROI_BOX_HEAD")
ROI_BOX_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""
ROI_BOX_HEAD_OUTPUTS_LAYER_REGISTRY = Registry("ROI_BOX_HEAD_OUTPUTS_LAYER")

def build_roi_heads_hook(cfg, *args,**kwargs):
    name = cfg.MODEL.ROI_HEADS.HOOK
    if len(name) > 0:
        if ";" in name:
            names = name.split(';')
            models = []
            for nm in names:
                models.append(ROI_HEADS_HOOK.get(nm)(cfg,*args,**kwargs))
            return WModelList(models,cfg,*args,**kwargs)
        else:
            return ROI_HEADS_HOOK.get(name)(cfg,*args,**kwargs)
    else:
        return None
    
def build_box_head(cfg, *args,**kwargs):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, *args,**kwargs)

def build_box_outputs_layer(cfg, *args,**kwargs):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.OUTPUTS_LAYER
    return ROI_BOX_HEAD_OUTPUTS_LAYER_REGISTRY.get(name)(cfg, *args,**kwargs)

def build_roi_heads(cfg, *args,**kwargs):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, *args,**kwargs)

build_outputs = _build_outputs
