from thirdparty.registry import Registry

BBDNET_MODEL = Registry("BBDNET")

def build_bbdnet(name,*args,**kwargs):
    return BBDNET_MODEL.get(name)(*args,**kwargs)
