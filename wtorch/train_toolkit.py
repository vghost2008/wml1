import os
import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from functools import partial
import torch.nn as nn
import time
import inspect

class WarmupCosLR(_LRScheduler):
    def __init__(self,optimizer, warmup_total_iters=1000,total_iters=120000,warmup_lr_start=1e-6,min_lr_ratio=0.01,last_epoch=-1, verbose=False):
        self.warmup_lr_start = warmup_lr_start
        self.warmup_total_iters = warmup_total_iters
        self.total_iters = total_iters
        self.min_lr_ratio = min_lr_ratio
        #init after member value initialized
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [self.__get_lr(x) for x in self.base_lrs]

    def __get_lr(self,lr):
        min_lr = lr * self.min_lr_ratio
        warmup_lr_start = self.warmup_lr_start
        iters = self.last_epoch
        warmup_total_iters = self.warmup_total_iters
        total_iters = self.total_iters

        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(
                iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0
                    + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters)
            )
            )
        return lr

class WarmupStepLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, step_size, gamma=0.1, warmup_total_iters=1000,total_iters=120000,warmup_lr_start=0,last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            print("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        iters = self.last_epoch
        warmup_total_iters = self.warmup_total_iters
        if iters <= warmup_total_iters:
            return [self.__get_warmup_lr(x) for x in self.base_lrs]
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]

    def __get_warmup_lr(self, lr):
        warmup_lr_start = self.warmup_lr_start
        iters = self.last_epoch
        warmup_total_iters = self.warmup_total_iters

        lr = (lr - warmup_lr_start) * pow(
            iters / float(warmup_total_iters), 2
        ) + warmup_lr_start
        return lr

def grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == math.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def simple_split_parameters(model,filter=None):
    pg0, pg1, pg2 = [], [], []
    parameters_set = set()
    print(f"------------------------------------------")
    for k, v in model.named_modules():
        if filter is not None and not(filter(k,v)):
            continue
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            if v.bias.requires_grad is False:
                print(f"{k}.bias requires grad == False, skip.")
            else:
                pg2.append(v.bias)  # biases
                parameters_set.add(k+".bias")
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            if v.weight.requires_grad is False:
                print(f"{k}.weight requires grad == False, skip.")
            else:
                pg0.append(v.weight)  # no decay
                parameters_set.add(k+".weight")
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            if v.weight.requires_grad is False:
                print(f"{k}.weight requires grad == False, skip.")
            else:
                pg1.append(v.weight)  # apply decay
                parameters_set.add(k+".weight")
        for k1,p in v.named_parameters(recurse=False):
            if k1 in ["weight","bias"]:
                continue
            if p.requires_grad == False:
                print(f"{k}.{k1} requires grad == False, skip.")
                continue
            if "weight" in k:
                pg1.append(p)
                parameters_set.add(k+f".{k1}")
            elif "bias" in k:
                pg2.append(p)
                parameters_set.add(k+f".{k1}")
            else:
                if p.ndim>1:
                    pg1.append(p)
                else:
                    pg2.append(p)
                parameters_set.add(k+f".{k1}")

    print(f"------------------------------------------")
    for k,p in model.named_parameters():
        if p.requires_grad == False:
            continue
        if k not in parameters_set:
            print(f"ERROR: {k} not in any parameters set.")
    #batch norm weight, weight, bias
    return pg0,pg1,pg2

def freeze_model(model,freeze_bn=True):
    if freeze_bn:
        model.eval()
    for name, param in model.named_parameters():
        print(name, param.size(), "freeze")
        param.requires_grad = False

def defrost_model(model,defrost_bn=True):
    if defrost_bn:
        model.train()
    for name, param in model.named_parameters():
        print(name, param.size(), "defrost")
        param.requires_grad = True

def __set_bn_momentum(m,momentum=0.1):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.momentum = momentum

def __fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def freeze_bn(model):
    model.apply(__fix_bn)

def set_bn_momentum(model,momentum):
    fn = partial(__set_bn_momentum,momentum=momentum)
    model.apply(fn)

def get_gpus_str(gpus):
    gpus_str = ""
    for g in gpus:
        gpus_str += str(g) + ","
    gpus_str = gpus_str[:-1]

    return gpus_str

def show_model_parameters_info(net):
    print("Training parameters.")
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name, list(param.size()), 'unfreeze')
    print("Not training parameters.")
    for name, param in net.named_parameters():
        if not param.requires_grad:
            print(name, list(param.size()), 'freeze')

def get_total_and_free_memory_in_Mb(cuda_device):
    devices_info_str = os.popen(
        "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader"
    )
    devices_info = devices_info_str.read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(",")
    return int(total), int(used)


def occupy_mem(cuda_device, mem_ratio=0.9):
    """
    pre-allocate gpu memory for training to avoid memory Fragmentation.
    """
    total, used = get_total_and_free_memory_in_Mb(cuda_device)
    max_mem = int(total * mem_ratio)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x
    time.sleep(5)

def isfinite_hook(module,fea_in,fea_out):
    if isinstance(fea_in,(tuple,list)):
        if len(fea_in)==1:
            fea_in = fea_in[0]
        elif len(fea_in)==0:
            return None
    #if not torch.all(torch.isfinite(fea_in)):
        #return None
    if not torch.all(torch.isfinite(fea_out)):
        print("Find NaN or infininite")
        print(f"{inspect.stack()}")
        print(f"Input : {torch.min(fea_in).item(),torch.max(fea_in).item(),torch.mean(fea_in).item()}")
        print(f"Output: {torch.min(fea_out).item(),torch.max(fea_out).item(),torch.mean(fea_out).item()}")
        for name, param in module.named_parameters():
            print(f"{name}: {torch.min(param).item(),torch.max(param).item(),torch.mean(param).item()}")

def islarge(x,max_v=65535):
    return torch.any(torch.abs(x)>max_v)

def islarge_hook(module,fea_in,fea_out):
    if isinstance(fea_in,(tuple,list)):
        if len(fea_in)==1:
            fea_in = fea_in[0]
        elif len(fea_in)==0:
            return None
    #if islarge(fea_in):
        #return None
    if islarge(fea_out):
        print("Find large value")
        print(f"{inspect.stack()}")
        print(f"Input : {torch.min(fea_in).item(),torch.max(fea_in).item(),torch.mean(fea_in).item()}")
        print(f"Output: {torch.min(fea_out).item(),torch.max(fea_out).item(),torch.mean(fea_out).item()}")
        for name, param in module.named_parameters():
            print(f"{name}: {torch.min(param).item(),torch.max(param).item(),torch.mean(param).item()}")

def register_forward_hook(net,hook):
    nr = 0
    for module in net.children():
        register_forward_hook(module,hook)
        nr += 1
    if nr == 0:
        net.register_forward_hook(hook=hook)
