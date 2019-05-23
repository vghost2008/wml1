#coding=utf-8
import os
import numpy as np
import os
import tensorflow as tf
import shutil
import scipy
import random
from tensorflow.contrib.slim.python.slim.data import parallel_reader
import time
from functools import wraps
import random

slim = tf.contrib.slim


def convert_to_nparray(value,shape=None,dtype=np.float32):
    if shape is None:
        return np.array(value)
    res = np.ndarray(shape=shape,dtype=dtype)
    assign_nparray(res,value)
    return res

def assign_nparray(array,value):
    shape = array.shape
    size = shape[0]
    if len(shape) == 1:
        for i in range(size):
            array[i] = value[i]
    else:
        for i in range(size):
            assign_nparray(array[i],value[i])

def reshape_list(l, shape=None):
    r = []
    if shape is None:
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i+s])
            i += s
    return r

def add_variables_summaries(learning_rate):
    summaries = []
    for variable in slim.get_model_variables():
        summaries.append(tf.summary.histogram(variable.op.name, variable))
    summaries.append(tf.summary.scalar('training/Learning Rate', learning_rate))
    return summaries


def update_model_scope(var, old_scope, new_scope):
    return var.op.name.replace(old_scope,new_scope)

def resize_list(in_data,size,default_value=None):
    if len(in_data)==size:
        return in_data
    if len(in_data)>size:
        return in_data[:size]
    in_data.extend([default_value]*(size-len(in_data)))
    return in_data

def get_filenames_in_dir(dir_path,suffix=None,prefix=None):
    if suffix is not None:
        suffix = suffix.split(";;")
    if prefix is not None:
        prefix = prefix.split(";;")

    def check_file(filename):
        is_suffix_good = False
        is_prefix_good = False
        if suffix is not None:
            for s in suffix:
                if filename.endswith(s):
                    is_suffix_good = True
                    break
        else:
            is_suffix_good = True
        if prefix is not None:
            for s in prefix:
                if filename.startswith(s):
                    is_prefix_good = True
                    break
        else:
            is_prefix_good = True

        return is_prefix_good and is_suffix_good
    res=[]
    for dir_path,_,files in os.walk(dir_path):
        for file in files:
            if suffix is not None or prefix is not None:
                if check_file(file):
                    res.append(file)
            else:
                res.append(file)
    res.sort()
    return res

def recurse_get_filepath_in_dir(dir_path,suffix=None,prefix=None):
    if suffix is not None:
        suffix = suffix.split(";;")
    if prefix is not None:
        prefix = prefix.split(";;")
    def check_file(filename):
        is_suffix_good = False
        is_prefix_good = False
        if suffix is not None:
            for s in suffix:
                if filename.endswith(s):
                    is_suffix_good = True
                    break
        else:
            is_suffix_good = True
        if prefix is not None:
            for s in prefix:
                if filename.startswith(s):
                    is_prefix_good = True
                    break
        else:
            is_prefix_good = True

        return is_prefix_good and is_suffix_good

    res=[]
    for dir_path,_,files in os.walk(dir_path):
        for file in files:
            if suffix is not None or prefix is not None:
                if check_file(file):
                    res.append(os.path.join(dir_path, file))
            else:
                res.append(os.path.join(dir_path,file))
    res.sort()
    return res

def recurse_get_filepath_in_dirs(dirs_path,suffix=None,prefix=None):
    files = []
    for dir in dirs_path:
        files.extend(recurse_get_filepath_in_dir(dir,suffix=suffix,prefix=prefix))
    files.sort()
    return files

def get_dirs(dir,subdirs):
    dirs=[]
    for sd in subdirs:
        dirs.append(os.path.join(dir,sd))
    return dirs

def _to_chinese_num(i,numbers,unites):
    j = i%10
    i = i/10
    res = numbers[j]
    if unites[0] is not None and len(unites)>0:
        res = res+unites[0]
    if i>0:
        return _to_chinese_num(i,numbers,unites[1:])+res
    else:
        return res

def to_chinese_num(i):
    if i==0:
        return "零"
    unites=[None,"十","百","千","万","十万","千万","亿","十亿"]
    numbers=["","一","二","三","四","五","六","七","八","九"]
    res = _to_chinese_num(i,numbers,unites)
    if res.startswith("一十"):
        res = res.decode("utf-8")
        res = res[1:]
        res = res.encode("utf-8")
    return res

def copy_and_rename_file(input_dir,output_dir,input_suffix=".jpg",out_name_pattern="IMG_%04d.jpg",begin_index=1):
    inputfilenames = recurse_get_filepath_in_dir(input_dir,suffix=input_suffix)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    i = begin_index
    for file in inputfilenames:
        new_path = os.path.join(output_dir,out_name_pattern%(i))
        shutil.copyfile(file,new_path)
        print("copy %s to %s.\n"%(file,new_path))
        i = i+1
    print("Copy finish.")

def safe_copy(src_file,dst_file):
    if os.path.exists(dst_file) and os.path.isdir(dst_file):
        dst_file = os.path.join(dst_file,os.path.basename(src_file))
        safe_copy(src_file,dst_file)
        return


    r_dst_file = dst_file
    if os.path.exists(r_dst_file):
        r_base_name = base_name(dst_file)
        r_suffix = suffix(dst_file)
        dst_dir = os.path.dirname(dst_file)
        index = 1
        while os.path.exists(r_dst_file):
            r_dst_file = os.path.join(dst_dir,r_base_name+f"_{index:02}."+r_suffix)
            index += 1

    shutil.copy(src_file,r_dst_file)


def base_name(v):
    base_name = os.path.basename(v)
    index = base_name.rfind(".")
    if -1 == index:
        return base_name
    else:
        return base_name[:index]

def suffix(v):
    base_name = os.path.basename(v)
    index = base_name.rfind(".")
    if -1 == index:
        return base_name
    else:
        return base_name[index+1:]

def npcrop_and_resize(image, box, crop_size):
    shape = image.shape
    if not isinstance(box,np.ndarray):
        box = np.array(box)
    ymin = int(box[0]*(shape[0]-1))
    ymax = int(box[2]*(shape[0]-1))
    xmin = int(box[1]*(shape[1]-1))
    xmax = int(box[3]*(shape[1]-1))

    if len(shape)==2:
        image = image[ymin:ymax,xmin:xmax]
    else:
        image = image[ymin:ymax,xmin:xmax,:]

    return npimresize(image,crop_size)

def npimresize(image,size):
    shape = image.shape
    if len(shape)==3 and shape[2]==1:
        image = np.squeeze(image,axis=2)
    image = scipy.misc.imresize(image,size)

    if len(image.shape)<len(shape):
        image = np.expand_dims(image,axis=2)

    return image

def home_dir(sub_path=None):
    if sub_path is None:
        return os.path.expandvars('$HOME')
    else:
        return os.path.join(os.path.expandvars('$HOME'),sub_path)

'''
suffix: suffix name without dot
'''
def change_suffix(path,suffix):
    dir_path = os.path.dirname(path)
    return os.path.join(dir_path,base_name(path)+"."+suffix)

def show_member(obj,name=None):
    if name is not None:
        print("Show %s."%(name))

    for name,var in vars(obj).items():
        print("%s : "%(name),var)

def show_list(values):
    if values is None:
        return
    if isinstance(values,str):
        return show_list([values])
    print("[")
    for v in values:
        print(v)
    print("]")
def show_dict(values):
    print("[")
    for k,v in values.items():
        print(k,"->",v)
    print("]")

def nparray2str(value,split=",",format="{}"):
    if not isinstance(value,np.ndarray):
        value = np.array(value)
    ndims = len(value.shape)
    if ndims == 1:
        r_str = "["
        for x in value[:-1]:
            r_str+=format.format(x)+split
        r_str+=format.format(value[-1])+"]"
        return r_str
    else:
        r_str = "["
        for x in value[:-1]:
            r_str+=nparray2str(x,split=split,format=format)+split
        r_str+=nparray2str(value[-1],split=split,format=format)+"]\n"
        return r_str

def show_nparray(value,name=None,split=",",format="{}"):
    if name is not None:
        print(name)
    print(nparray2str(value,split=split,format=format))

class ExperienceBuffer():
    def __init__(self, buffer_size = 100000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0: (len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        if not isinstance(self.buffer[0],np.ndarray):
            data = random.sample(self.buffer,size)
            data = list(zip(*data))
            return [np.array(list(x)) for x in data]
        else:
            return np.reshape(np.array(random.sample(self.buffer, size)), [size]+self.shape)


class CycleBuffer:
    def __init__(self,cap=5):
        self.cap = cap
        self.buffer = []
    def append(self,v):
        self.buffer.append(v)
        l = len(self.buffer)
        if l>self.cap:
            self.buffer = self.buffer[l-self.cap:]

    def __getitem__(self, slice):
        return self.buffer[slice]

    def __len__(self):
        return len(self.buffer)

def remove_hiden_file(files):
    res = []
    for file in files:
        if os.path.basename(file).startswith("."):
            continue
        res.append(file)

    return res

def reduce_prod(x):
    if len(x)==0:
        return 0
    elif len(x)==1:
        return x[0]
    res = x[0]
    for v in x[1:]:
        res *= v

    return res

def any(iterable,v=None):
    if v is None:
        for value in iterable:
            if value is None:
                return True
        return False
    else:
        t = type(v)
        for value in iterable:
            if isinstance(t,value) and v==value:
                return True

        return False

def all(iterable,v=None):
    if v is None:
        for value in iterable:
            if value is not None:
                return False
        return True
    else:
        t = type(v)
        for value in iterable:
            if (not isinstance(t,value)) or v!=value:
                return False

        return True

def gather(data,indexs):
    res_data = []
    
    for d in indexs:
        res_data.append(data[d])
    
    return res_data

class TimeThis():
    def __init__(self,name="TimeThis"):
        self.begin_time = 0.
        self.name = name

    def __enter__(self):
        self.begin_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"{self.name}: total time {time.time()-self.begin_time}.")


def time_this(func):
    @wraps(func)
    def wraps_func(*args,**kwargs):
        begin_t = time.time()
        res = func(*args,**kwargs)
        print(f"Time cost {time.time()-begin_t}.")
        return res
    return wraps_func

class MDict(dict):
    def __init__(self, *args, **kw):
        dict.__init__(self, *args, **kw)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

def list_to_2dlist(data,size):
    data_nr = len(data)
    if data_nr<size:
        return [data]
    res = []
    index = 0
    while index<data_nr:
       end_index = min(index+size,data_nr)
       res.append(data[index:end_index])
       index = end_index
    return res

def random_uniform(minmaxs):
    res = []
    for min,max in minmaxs:
        res.append(random.uniform(min,max))
    return res

def random_uniform_indict(minmaxs):
    res = {}
    for key,(min,max) in minmaxs.items():
        res[key] = random.uniform(min,max)
    return res
