import torch
import numpy as np
from torch._six import queue, container_abcs, string_classes
from collections import Iterable
import torch.nn.functional as F
import random
import sys
from functools import wraps

def unnormalize(x:torch.Tensor,mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0]):
    if len(x.size())==4:
        scale = np.reshape(np.array(std,dtype=np.float32),[1,3,1,1])
        offset = np.reshape(np.array(mean,dtype=np.float32),[1,3,1,1])
    elif len(x.size())==5:
        scale = np.reshape(np.array(std, dtype=np.float32), [1, 1,3, 1, 1])
        offset = np.reshape(np.array(mean, dtype=np.float32), [1,1, 3, 1, 1])
    elif len(x.size())==3:
        scale = np.reshape(np.array(std, dtype=np.float32), [3, 1, 1])
        offset = np.reshape(np.array(mean, dtype=np.float32), [3, 1, 1])

    x = x*scale+offset
    return x

def normalize(x:torch.Tensor,mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0]):
    if len(x.size())==4:
        scale = np.reshape(np.array(std,dtype=np.float32),[1,3,1,1])
        offset = np.reshape(np.array(mean,dtype=np.float32),[1,3,1,1])
    elif len(x.size())==5:
        scale = np.reshape(np.array(std, dtype=np.float32), [1, 1,3, 1, 1])
        offset = np.reshape(np.array(mean, dtype=np.float32), [1,1, 3, 1, 1])
    elif len(x.size())==3:
        scale = np.reshape(np.array(std, dtype=np.float32), [3, 1, 1])
        offset = np.reshape(np.array(mean, dtype=np.float32), [3, 1, 1])

    offset = torch.from_numpy(offset).to(x.device)
    scale = torch.from_numpy(scale).to(x.device)
    x = (x-offset)/scale
    return x

def npnormalize(x:np.ndarray,mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0]):
    if len(x.shape)==4:
        scale = np.reshape(np.array(std,dtype=np.float32),[1,3,1,1])
        offset = np.reshape(np.array(mean,dtype=np.float32),[1,3,1,1])
    elif len(x.shape)==5:
        scale = np.reshape(np.array(std, dtype=np.float32), [1, 1,3, 1, 1])
        offset = np.reshape(np.array(mean, dtype=np.float32), [1,1, 3, 1, 1])
    elif len(x.shape)==3:
        scale = np.reshape(np.array(std, dtype=np.float32), [3, 1, 1])
        offset = np.reshape(np.array(mean, dtype=np.float32), [3, 1, 1])

    x = (x.astype(np.float32)-offset)/scale

    return x

def remove_prefix_from_state_dict(state_dict,prefix="module."):
    res = {}
    for k,v in state_dict.items():
        if k.startswith(prefix):
            k = k[len(prefix):]
        res[k] = v
    return res

def forgiving_state_restore(net, loaded_dict,verbose=False):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """

    if 'state_dict' in loaded_dict:
        loaded_dict = loaded_dict['state_dict']
    if hasattr(net,'module'):
        net = net.module
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    used_loaded_dict_key = []
    for k in net_state_dict:
        new_k = k
        if new_k in loaded_dict and net_state_dict[k].size() == loaded_dict[new_k].size():
            new_loaded_dict[k] = loaded_dict[new_k]
        elif (not k.startswith('module.')) and 'module.'+k in loaded_dict and net_state_dict[k].size() == loaded_dict['module.'+new_k].size():
            new_loaded_dict[k] = loaded_dict['module.'+new_k]
            used_loaded_dict_key.append('module.'+new_k)
        else:
            print("Skipped loading parameter {}".format(k))

    print(f"---------------------------------------------------")
    for k in loaded_dict:
        if k not in new_loaded_dict and k not in used_loaded_dict_key:
            print(f"Skip {k} in loaded dict")
    if verbose:
        print(f"---------------------------------------------------")
        for k in new_loaded_dict:
            print(f"Load {k}")
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    sys.stdout.flush()
    return net

def sequence_mask(lengths,maxlen=None,dtype=torch.bool):
    if not isinstance(lengths,torch.Tensor):
        lengths = torch.from_numpy(np.array(lengths))
    if maxlen is None:
        maxlen = lengths.max()
    if len(lengths.shape)==1:
        lengths = torch.unsqueeze(lengths,axis=-1)
    matrix = torch.arange(maxlen,dtype=lengths.dtype)[None,:]
    mask = matrix<lengths
    return mask


class TraceAmpWrape(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                return self.model(x)

def get_tensor_info(tensor):
    tensor = tensor.detach().cpu().to(torch.float32)
    return torch.mean(tensor).item(),torch.min(tensor).item(),torch.max(tensor).item(),torch.std(tensor).item()

def merge_imgs_heatmap(imgs,heat_map,scale=1.0,alpha=0.4,channel=None,min=None,max=None):
    if not isinstance(heat_map,torch.Tensor):
        heat_map = torch.from_numpy(heat_map)
    if not isinstance(imgs,torch.Tensor):
        imgs = torch.from_numpy(imgs)
    if min is None:
        min = torch.min(heat_map)
    else:
        heat_map = torch.maximum(heat_map,torch.Tensor([min]))

    if max is None:
        max = torch.max(heat_map)
    else:
        heat_map = torch.minimum(heat_map,torch.Tensor([max]))
    heat_map = (heat_map-min)*scale/(max-min+1e-8)
    if channel is not None and heat_map.shape[channel]==1:
        t_zeros = torch.zeros_like(heat_map)
        heat_map = torch.cat([heat_map,t_zeros,t_zeros],dim=channel)
    new_imgs = imgs*(1-alpha)+heat_map*alpha
    mask = heat_map>(scale*0.01)
    #imgs = torch.where(mask,new_imgs,imgs)
    imgs = new_imgs
    return imgs

def module_parameters_numel(net,only_training=False):
    total = 0
    for param in net.parameters():
        if only_training and param.requires_grad or not only_training:
            total += torch.numel(param)
    return total


def concat_datas(datas,dim=0):
    if isinstance(datas[0], container_abcs.Mapping):
        new_data = {}
        for k,v in datas[0].items():
            new_data[k] = [v]
        for data in datas[1:]:
            for k,v in data.items():
                new_data[k].append(v)
        keys = list(new_data.keys())
        for k in keys:
            new_data[k] = torch.cat(new_data[k],dim=dim)
        return new_data

    if torch.is_tensor(datas[0]):
        return torch.cat(datas,dim=dim)
    elif isinstance(datas[0],Iterable):
        res = []
        try:
            for x in zip(*datas):
                if torch.is_tensor(x[0]):
                    res.append(torch.cat(x,dim=dim))
                else:
                    res.append(concat_datas(x))
        except Exception as e:
            print(e)
            for i,x in enumerate(datas):
                print(i,type(x),x)
            print(f"--------------------------")
            for i,x in enumerate(datas):
                print(i,type(x))
            sys.stdout.flush()
            raise e
        return res
    else:
        return torch.cat(datas,dim=dim)

def get_model(model):
    if hasattr(model, "module"):
        model = model.module
    return model

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

'''
fea:[B,C,H,W]
size:(w,h)
'''
CENTER_PAD = 0
RANDOM_PAD = 1
TOPLEFT_PAD = 2
def pad_feature(fea, size, pad_value=0, pad_type=TOPLEFT_PAD, return_pad_value=False):
    '''
    pad_type: 0, center pad
    pad_type: 1, random pad
    pad_type: 2, topleft_pad
    '''
    w = fea.shape[-1]
    h = fea.shape[-2]
    if pad_type == 0:
        if h < size[1]:
            py0 = (size[1] - h) // 2
            py1 = size[1] - h - py0
        else:
            py0 = 0
            py1 = 0
        if w < size[0]:
            px0 = (size[0] - w) // 2
            px1 = size[0] - w - px0
        else:
            px0 = 0
            px1 = 0
    elif pad_type == 1:
        if h < size[1]:
            py0 = random.randint(0, size[1] - h)
            py1 = size[1] - h - py0
        else:
            py0 = 0
            py1 = 0
        if w < size[0]:
            px0 = random.randint(0, size[0] - w)
            px1 = size[0] - w - px0
        else:
            px0 = 0
            px1 = 0
    elif pad_type == 2:
        if h < size[1]:
            py0 = 0
            py1 = size[1] - h - py0
        else:
            py0 = 0
            py1 = 0
        if w < size[0]:
            px0 = 0
            px1 = size[0] - w - px0
        else:
            px0 = 0
            px1 = 0

    fea = F.pad(fea, [px0, px1,py0,py1], "constant", pad_value)

    if return_pad_value:
        return fea, px0, px1, py0, py1
    return fea

def split_forward_batch32(func):
    @wraps(func)
    def wrapper(self, data):
        step = 32
        res = []
        cur_idx = 0
        while cur_idx<data.shape[0]:
            ret_val = func(self, data[cur_idx:cur_idx+step])
            cur_idx += step
            res.append(ret_val)
        if len(res)==1:
            return res[0]
        if torch.is_tensor(res[0]):
            return torch.cat(res,dim=0)
        else:
            return np.concatenate(res,axis=0)
    return wrapper