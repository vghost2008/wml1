from thirdparty.registry import Registry

MOT_HEAD = Registry("MOTHEADS")

def build_MOT_head(name,*args,**kwargs):
    return MOT_HEAD.get(name)(*args,**kwargs)