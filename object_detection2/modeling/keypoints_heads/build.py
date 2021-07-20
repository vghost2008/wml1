from thirdparty.registry import Registry

KEYPOINTS_HEAD = Registry("KeyPointsHead")

def build_keypoints_head(name,*args,**kwargs):
    return KEYPOINTS_HEAD.get(name)(*args,**kwargs)