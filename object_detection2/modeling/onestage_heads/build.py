from thirdparty.registry import Registry

RETINANET_HEAD = Registry("RetinaNetHead")

def build_retinanet_head(name,*args,**kwargs):
    return RETINANET_HEAD.get(name)(*args,**kwargs)
