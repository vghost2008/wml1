import torch
import numpy as np

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
