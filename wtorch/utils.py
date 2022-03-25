import torch
import numpy as np
import sys

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
