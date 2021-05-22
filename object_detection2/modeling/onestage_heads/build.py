from thirdparty.registry import Registry

RETINANET_HEAD = Registry("RetinaNetHead")
ONESTAGE_HEAD = Registry("OneStageHead")

def build_retinanet_head(name,*args,**kwargs):
    return RETINANET_HEAD.get(name)(*args,**kwargs)

def build_onestage_head(name,*args,**kwargs):
    return ONESTAGE_HEAD.get(name)(*args,**kwargs)
