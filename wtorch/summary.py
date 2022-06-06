import torch
from wsummary import _draw_text_on_image
from collections import Iterable
import object_detection2.visualization as odv
import random
import numpy as np
import object_detection2.bboxes as odb
#import tensorboardX as tb
#tb.SummaryWriter.add_video()
#tb.add_images

def log_all_variable(tb,net:torch.nn.Module,global_step):
    try:
        for name,param in net.named_parameters():
            if 'bias' in name:
                name = "BIAS/"+name
            elif "." in name:
                name = name.replace(".","/",1)
            if param.numel()>1:
                tb.add_histogram(name,param,global_step)
            else:
                tb.add_scalar(name,param,global_step)

        data = net.state_dict()
        for name in data:
            if "running" in name:
                param = data[name]
                if param.numel()>1:
                    tb.add_histogram("BN/"+name,param,global_step)
                else:
                    tb.add_scalar("BN/"+name,param,global_step)
    except Exception as e:
        print("ERROR:",e)

def log_basic_info(tb,name,value:torch.Tensor,global_step):
    if value.numel()>1:
        min_v = torch.min(value)
        max_v = torch.max(value)
        mean_v = torch.mean(value)
        std_v = torch.std(value)
        tb.add_scalar(name+"/min",min_v,global_step)
        tb.add_scalar(name+"/max",max_v,global_step)
        tb.add_scalar(name+"/mean",mean_v,global_step)
        tb.add_scalar(name+"/std",std_v,global_step)
    else:
        tb.add_scalar(name,value,global_step)

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

def log_feature_map(tb,name,tensor,global_step,random_index=True):
    '''
    tensor: [B,C,H,W]
    '''
    if isinstance(tensor,torch.Tensor):
        tensor = tensor.cpu().detach().numpy()

    if random_index:
        i = random.randint(0,tensor.shape[0]-1)
    else:
        i = 0
    data = tensor[i]
    data = np.expand_dims(data,axis=1)
    min = np.min(data)
    max = np.max(data)
    data = (data-min)/(max-min+1e-8)
    tb.add_images(name,data,global_step)

def try_log_rgb_feature_map(tb,name,tensor,global_step,random_index=True,min_upper_bounder=None,max_lower_bounder=None):
    if isinstance(tensor,torch.Tensor):
        tensor = tensor.cpu().detach().numpy()

    if random_index:
        i = random.randint(0,tensor.shape[0]-1)
    else:
        i = 0
    C = tensor.shape[1] 
    data = tensor[i]
    min = np.min(data)
    if min_upper_bounder is not None:
        min = np.minimum(min,min_upper_bounder)
    max = np.max(data)
    if max_lower_bounder is not None:
        max = np.maximum(max,max_lower_bounder)
    data = (data-min)/(max-min+1e-8)
    if C>3:
        data = np.expand_dims(data,axis=1)
        tb.add_images(name,data,global_step)
    else:
        if C==2:
            _,H,W = data.shape
            zeros = np.zeros([1,H,W],dtype=data.dtype)
            data = np.concatenate([data,zeros],axis=0)
        tb.add_image(name,data,global_step)

    
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
    if isinstance(video,torch.Tensor):
        video = video.numpy()
    video = video.transpose(0,1,3,4,2)
    video = np.ascontiguousarray(video)
    if label is not None:
        for i in range(video.shape[0]):
            l = label[i]
            for j in range(video.shape[1]):
                _draw_text_on_image(video[i,j],l,font_scale=font_scale)
    #video (N,T,H,W,C)
    video = video.transpose(0,1,4,2,3)
    tb.add_video(name,video,global_step)

def log_mask(tb,tag,images,masks,step,color_map,img_min=None,img_max=None,ignore_label=255,max_images=4,save_raw_imgs=False):
    images = images[:max_images]
    masks = masks[:max_images]
    if img_min or img_max is None:
        img_min = torch.min(images)
        img_max = torch.max(images)
    images = (images-img_min)*255/(img_max-img_min+1e-8)
    images = torch.clip(images,0,255)
    images = images.to(torch.uint8)
    images = images.permute(0,2,3,1).cpu().numpy()
    masks = masks.cpu().numpy()
    res_imgs = []
    for img,msk in zip(images,masks):
        r_img = odv.draw_semantic_on_image(img,msk,color_map,ignored_label=ignore_label)
        res_imgs.append(r_img)
    res_images = np.stack(res_imgs,axis=0)
    tb.add_images(tag,res_images,step,dataformats='NHWC')
    if save_raw_imgs:
        tb.add_images(tag+"_raw",images,step,dataformats='NHWC')


def log_optimizer(tb,optimizer,step):
    for i,data in enumerate(optimizer.param_groups):
        name = f"optimizer/{i}_{len(data['params'])}"
        tb.add_scalar(name+"_lr",data['lr'],step)
        tb.add_scalar(name+"_wd",data['weight_decay'],step)


def log_imgs_with_bboxes(tb,name,imgs,targets,step,max_imgs=None):
    '''
    imgs: [B,C,H,W] [0-255]
    targets: [B,5] [label,x0,y0,x1,y1]
    '''
    if max_imgs is not None and max_imgs>0:
        imgs = imgs[:max_imgs]
        targets = targets[:max_imgs]
    
    if torch.is_tensor(imgs):
        imgs = imgs.detach().cpu().numpy().astype(np.uint8)
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    
    bboxes = targets[...,1:5]
    labels = targets[...,0].astype(np.int32)
    bboxes = odb.npchangexyorder(bboxes)

    res_imgs = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = np.ascontiguousarray(np.transpose(img,[1,2,0]))
        cur_bboxes = bboxes[i]
        cur_labels = labels[i]
        bboxes_nr = np.count_nonzero(np.sum(cur_bboxes,axis=-1)>0)
        cur_bboxes = cur_bboxes[:bboxes_nr]
        cur_labels = cur_labels[:bboxes_nr]
        img = odv.draw_bboxes(img,cur_labels,bboxes=cur_bboxes,is_relative_coordinate=False)
        res_imgs.append(img)
    
    res_imgs = np.array(res_imgs)
    res_imgs = np.ascontiguousarray(res_imgs)
    tb.add_images(name,res_imgs,step,dataformats='NHWC')

