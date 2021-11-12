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

def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """

    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        new_k = k
        if new_k in loaded_dict and net_state_dict[k].size() == loaded_dict[new_k].size():
            new_loaded_dict[k] = loaded_dict[new_k]
        elif (not k.startswith('module.')) and 'module.'+k in loaded_dict and net_state_dict[k].size() == loaded_dict['module.'+new_k].size():
            new_loaded_dict[k] = loaded_dict['module.'+new_k]
        else:
            print("Skipped loading parameter {}".format(k))
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
