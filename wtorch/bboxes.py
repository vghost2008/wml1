import torch

def cxywh2xy(bboxes):
    cxy = bboxes[...,:2]
    hwh = bboxes[...,2:]/2
    minxy = cxy-hwh
    maxxy = cxy+hwh
    return torch.cat([minxy,maxxy],dim=-1)

def xy2cxywh(bboxes):
    wh = bboxes[...,2:]-bboxes[...,:2]
    cxy = (bboxes[...,2:]+bboxes[...,:2])/2
    return torch.cat([cxy,wh],dim=-1)

def distored_boxes(bboxes:torch.Tensor,scale=[0.8,1.2],offset=0.2):
    bboxes = xy2cxywh(bboxes)
    cxy,wh = torch.split(bboxes,2,dim=-1)
    wh_scales = torch.rand(list(wh.shape),dtype=bboxes.dtype)*(scale[1]-scale[0])+scale[0]
    wh_scales = wh_scales.to(wh.device)
    wh = wh*wh_scales
    cxy_offset = torch.rand(list(cxy.shape),dtype=cxy.dtype)*offset
    cxy_offset = cxy_offset.to(cxy.device)
    cxy = cxy+cxy_offset
    bboxes = torch.cat([cxy,wh],axis=-1)
    bboxes = cxywh2xy(bboxes)
    bboxes = torch.nn.functional.relu(bboxes)
    return bboxes

