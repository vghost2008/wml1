import torch
from wsummary import _draw_text_on_image
from collections import Iterable
import numpy as np
#import tensorboardX as tb
#tb.SummaryWriter.add_video()

def log_all_variable(tb,net:torch.nn.Module,global_step):
    for name,param in net.named_parameters():
        if param.numel()>1:
            tb.add_histogram(name,param,global_step)
        else:
            tb.add_scalar(name,param,global_step)

    data =  net.state_dict()
    for name in data:
        if "running" in name:
            param = data[name]
            if param.numel()>1:
                tb.add_histogram("BN_"+name,param,global_step)
            else:
                tb.add_scalar("BN"+name,param,global_step)

def add_image_with_label(tb,name,image,label,global_step):
    label = str(label)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = _draw_text_on_image(image,label)
    image = image.transpose(2,0,1)
    tb.add_image(name,image,global_step)

def add_images_with_label(tb,name,image,label,global_step,font_scale=1.2):
    if isinstance(image,torch.Tensor):
        image = image.numpy()
    image = image.transpose(0,2,3,1)
    image = np.ascontiguousarray(image)
    if not isinstance(label,Iterable):
        label = str(label)
        image[0] = _draw_text_on_image(image[0], label,font_scale=font_scale)
    elif len(label) == 1:
        label = str(label[0])
        image[0] = _draw_text_on_image(image[0],label,font_scale=font_scale)
    elif len(label) == image.shape[0]:
        for i in range(len(label)):
            image[i] = _draw_text_on_image(image[i], str(label[i]),font_scale=font_scale)
    else:
        print(f"ERROR label {label}")
        return

    image = image.transpose(0,3,1,2)
    tb.add_images(name,image,global_step)

def add_video_with_label(tb,name,video,label,global_step,fps=4,font_scale=1.2):
    '''
    Args:
        tb:
        name:
        video:  (N, T, C, H, W)
        label:
        global_step:
        fps:
        font_scale:

    Returns:

    '''
    video = video.transpose(0,1,3,4,2)
    if isinstance(video,torch.Tensor):
        video = video.numpy()
    video = np.ascontiguousarray(video)
    for i in range(video.shape[0]):
        l = label[i]
        for j in range(video.shape[1]):
            _draw_text_on_image(video[i,j],l,font_scale=font_scale)
    #video (N,T,H,W,C)
    video = video.transpose(0,1,4,2,3)
    tb.add_video(name,video,global_step)
