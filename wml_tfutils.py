#coding=utf-8
from wtfop.wtfop_ops import set_value
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python import pywrap_tensorflow
import wml_utils as wmlu
import os
import numpy as np
import math
import time
import logging
import cv2
import wsummary
import basic_tftools as btf
from tensorflow.python.framework import graph_util
from functools import wraps
from collections import Iterable
from iotoolkit.transform import  distort_color as _distort_color


_HASH_TABLE_COLLECTION = "HASH_TABLE"
_MEAN_RGB = [123.15, 115.90, 103.06]

distort_color = _distort_color
isSingleValueTensor = btf.isSingleValueTensor
static_or_dynamic_map_fn = btf.static_or_dynamic_map_fn
variable_summaries = wsummary.variable_summaries
variable_summaries_v2 = wsummary.variable_summaries_v2
histogram = wsummary.histogram
histogram_or_scalar = wsummary.histogram_or_scalar
image_summaries = wsummary.image_summaries
_draw_text_on_image = wsummary._draw_text_on_image
image_summaries_with_label = wsummary.image_summaries_with_label
row_image_summaries = wsummary.row_image_summaries

def add_to_hash_table_collection(value):
    tf.add_to_collection(_HASH_TABLE_COLLECTION,value)

def get_hash_table_collection():
    return tf.get_collection(_HASH_TABLE_COLLECTION)


def parameterNum(argus):
    num = 0
    print(type(argus))
    if isinstance(argus,dict):
        argus = argus.values()
    for argu in argus:
        dim=1
        shape = argu.get_shape().as_list()
        for v in shape:
            dim *= v
        num += dim
    return num

def show_values(values,name=None,fn=print):
    string = ""
    if name is not None:
        string += name+"\n"
    for v in values:
        string += str(v)+"\n"
    fn(string)

def show_values_name(values,name=None,fn=print):
    string = ""
    if name is not None:
        string += name+"\n"
    for v in values:
        string += str(v.name)+"\n"
    fn(string)

def gather_in_axis_with_one_dim_indices(data,indices,axis=0):
    '''
    :param data: a tensor with more than one dims
    :param indices: one dim indeces
    :param axis:
    :return:
    example:
    data = [[1,3,2],[9,8,7]]
    indices = [1,2,0]
    res = [[3,2,1],[8,7,9]]
    '''

    assert data.get_shape().ndims<=1,"error indices dim."

    if axis == 0:
        return tf.gather(data,indices)
    indices = tf.reshape(indices,[-1])
    perm = range(len(data.get_shape().as_list()))
    perm[0] = axis
    perm[axis] = 0
    data = tf.transpose(data,perm=perm)
    data = tf.gather(data,indices)
    data = tf.transpose(data,perm)
    return data

def gather_in_axis_with_two_dim_indices(data,indices,axis=0):
    '''
    :param data: [batch_size,...], a tensor with more than one dims.
    :param indices: [batch_size,X], indices with exactly two dims.
    :param axis:
    example:
    data = [[1,3,2],[7,8,9]]
    indices = [[1,2,0],[2,1,0]]
    res = [[3,2,1],[9,8,7]]
    '''
    assert indices.get_shape().ndims ==2, "error indices dim."

    if axis == 0:
        return tf.gather(data, indices)
    if axis==1:
        data = tf.map_fn(lambda x:tf.gather(x[0],x[1]), elems=(data,indices),dtype=(data.dtype))
    else:
        perm = range(len(data.get_shape().as_list()))
        perm[1] = axis
        perm[axis] = 1
        data = tf.transpose(data, perm=perm)
        data = tf.map_fn(lambda x:tf.gather(x[0],x[1]), elems=(data,indices),dtype=(data.dtype))
        data = tf.transpose(data, perm)
    return data

def gather_in_axis(data,indices,axis=0):
    if axis == 0:
        return tf.gather(data,indices)
    if indices.get_shape().ndims<=1:
        return gather_in_axis_with_one_dim_indices(data,indices,axis)
    else:
        return gather_in_axis_with_two_dim_indices(data,indices,axis)
    return data


'''
'''
def apply_with_random_selector(x, func, num_cases):
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    '''
    只有当case==sel时func才会收到一个可用的tensor
    merge返回一个available tensor和index
    '''
    return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
            for case in range(num_cases)])[0]

def _ImageDimensions(image):
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(3).as_list()
        dynamic_shape = array_ops.unstack(array_ops.shape(image), 3)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

def resize_image(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
    with tf.name_scope('resize_image'):
        height, width, channels = _ImageDimensions(image)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size,
                                       method, align_corners)
        image = tf.reshape(image, tf.stack([size[0], size[1], channels]))
        return image

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


'''
将var作为图像记录
var:[batch_size,X]
'''
def scale_image_summaries(var,name,max_outputs=3,heigh=None):
    shape = var.get_shape().as_list()
    if heigh is None:
        heigh = shape[-1]/3
    var = tf.expand_dims(var,axis=1)
    var = tf.expand_dims(var,axis=3)
    var = tf.image.resize_nearest_neighbor(var,size=[heigh,shape[-1]])
    tf.summary.image(name,var,max_outputs=max_outputs)

def top_k_mask(value,k=1,shape=None):
    values, indics = tf.nn.top_k(value, k)
    indics, _ = tf.nn.top_k(-indics, k)
    if value.get_shape().is_fully_defined():
        res = tf.cast(tf.sparse_to_dense(-indics, value.shape, 1), tf.bool)
    else:
        shape = tf.shape(value)
        res = tf.cast(tf.sparse_to_dense(-indics, shape,1),tf.bool)
    if shape is None:
         return res
    else:
        return tf.reshape(res,shape)

def bottom_k_mask(value, k=1,shape=None):
    return top_k_mask(-value,k,shape)

'''
根据index指定的值在x的第二维中选择数据
index: (Y)
x:(Y,M,N,...) 
return:
x:(Y,N,...)
'''
def select_2thdata_by_index(x,index):
    if not isinstance(x,tf.Tensor):
        x = tf.convert_to_tensor(x)
    if not isinstance(index,tf.Tensor):
        index = tf.convert_to_tensor(index)
    if not x.get_shape().is_fully_defined() or not index.get_shape().is_fully_defined():
        return select_2thdata_by_index_v2(x,index)
    d_shape = index.get_shape().as_list()
    x_2th_size = x.get_shape().as_list()[1]
    range = tf.range(0, d_shape[0],dtype=tf.int32)
    range = tf.expand_dims(range, axis=1)
    index = tf.expand_dims(index, axis=1)
    if index.dtype is not tf.int32:
        index = tf.cast(index,tf.int32)
    d_masks = tf.concat(values=[range, index], axis=1)
    d_masks = tf.sparse_to_dense(d_masks, [d_shape[0], x_2th_size], 1)
    res = tf.boolean_mask(x, tf.cast(d_masks, tf.bool))
    return res

def select_2thdata_by_index_v2(x,index):
    '''
    handle with the situation which x or index's shape is not fully defined.
    :param x: (Y,M,N,...)
    :param index: (Y)
    :return: (Y,N,...)
    '''
    if not isinstance(x,tf.Tensor):
        x = tf.convert_to_tensor(x)
    if not isinstance(index,tf.Tensor):
        index = tf.convert_to_tensor(index)
    d_shape = tf.shape(index)
    x_2th_size = tf.shape(x)[1]
    range = tf.range(0, d_shape[0],dtype=tf.int32)
    range = tf.expand_dims(range, axis=1)
    index = tf.expand_dims(index, axis=1)
    if index.dtype is not tf.int32:
        index = tf.cast(index,tf.int32)
    d_masks = tf.concat(values=[range, index], axis=1)
    d_masks = tf.sparse_to_dense(d_masks, [d_shape[0], x_2th_size], 1)
    res = tf.boolean_mask(x, tf.cast(d_masks, tf.bool))
    return res

def select_2thdata_by_index_v3(x,index):
    '''
    handle with the situation which x or index's first two dim is not fully defined.
    :param x: (Y,M,N,...)
    :param index: (Y)
    :return: (Y,N,...)
    '''
    if not isinstance(x,tf.Tensor):
        x = tf.convert_to_tensor(x)
    if not isinstance(index,tf.Tensor):
        index = tf.convert_to_tensor(index)
    batch_size = x.get_shape().as_list()[0]
    old_shape = tf.shape(x)
    new_shape = [-1]+x.get_shape().as_list()[2:]
    x = tf.reshape(x,new_shape)
    res = tf.gather(x, tf.range(old_shape[0], dtype=tf.int32) * old_shape[1]+ index)
    if batch_size is not None:
        res = tf.reshape(res,[batch_size]+new_shape[1:])
    return res

def get_ckpt_file_path(path):
    if tf.gfile.IsDirectory(path):
        try:
            ckpt_state = tf.train.get_checkpoint_state(path)
            if ckpt_state is not None:
                return ckpt_state.model_checkpoint_path
            else:
                print("Error checkpoint state.")
                return None
        except tf.errors.OutOfRangeError as e:
            print("Cannot restore checkpoint:%s" % e)
            return None
    elif tf.gfile.Exists(path):
        return path
    #process the situation of path is a tensorflow check point file
    #like ../../tmp/tod_traindatav1/data.ckpt-3901
    dir_path = os.path.dirname(path)
    file_name = os.path.basename(path)

    if ".ckpt" not in file_name:
        return None

    files = wmlu.recurse_get_filepath_in_dir(dir_path)

    for f in files:
        f = os.path.basename(f)
        if f.startswith(file_name):
            return path

    return None

def get_variables_in_ckpt(file_path):
    reader = pywrap_tensorflow.NewCheckpointReader(file_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return list(var_to_shape_map.keys())

def get_variables_in_ckpt_in_dir(dir_path):
    file_path = get_ckpt_file_path(dir_path)
    return get_variables_in_ckpt(file_path)

def get_variables_dict_in_ckpt(file_path):
    reader = pywrap_tensorflow.NewCheckpointReader(file_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return var_to_shape_map

def get_variables_dict_in_ckpt_in_dir(dir_path):
    file_path = get_ckpt_file_path(dir_path)
    if file_path is None:
        return None
    return get_variables_dict_in_ckpt(file_path)

def get_variables_unrestored(restored_values,file_path,exclude_var=None):
    variables_in_ckpt = get_variables_in_ckpt(file_path)
    for value in restored_values:
        if value in variables_in_ckpt:
            variables_in_ckpt.remove(value)

    res_variable = list(variables_in_ckpt)
    if exclude_var is not None:
        scopes = [ scope.strip() for scope in exclude_var.split(",")]
        for scope in scopes:
            for v in variables_in_ckpt:
                if scope in v:
                    res_variable.remove(v)
    return res_variable

def get_variables_unrestoredv1(restored_values,exclude_var=None):
    all_variables = list(map(lambda x:x.name,tf.global_variables()))
    for i, v in enumerate(all_variables):
        index = v.find(':')
        if index > 0:
            all_variables[i] = all_variables[i][:index]
    for value in restored_values:
        if value in all_variables:
            all_variables.remove(value)

    res_variable = list(all_variables)
    if exclude_var is not None:
        scopes = [ scope.strip() for scope in exclude_var.split(",")]
        for scope in scopes:
            for v in all_variables:
                if scope in v:
                    res_variable.remove(v)
    return res_variable

def int64_feature(value):
    if not isinstance(value, list) and not isinstance(value,np.ndarray):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))



def floats_feature(value):
    if not isinstance(value, list) and not isinstance(value,np.ndarray):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    value = [v if isinstance(v,bytes) else v.encode("utf-8") for v in value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def bytes_vec_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def draw_points(points,image,color,size=3):
    pass


def merge(scopes=None):
    if scopes is None:
        return tf.summary.merge_all()
    scopes_list = [scope.strip() for scope in scopes.split(',')]
    summaries_list = []
    for scope in scopes_list:
        values = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        summaries_list.extend(values)

    return tf.summary.merge(summaries_list)

def merge_exclude(excludes=None):
    if excludes is None:
        return tf.summary.merge_all()
    vars = tf.get_collection(tf.GraphKeys.SUMMARIES)
    exclude_list = [exclude.strip() for exclude in excludes.split(',')]
    summaries_list = []
    for exclude in exclude_list:
        summaries_list = []
        for var in vars:
            if not var.name.startswith(exclude):
                summaries_list.append(var)
        vars = summaries_list

    #wmlu.show_list(summaries_list)

    return tf.summary.merge(summaries_list)

def join_scopes(scope,subscopes):
    if isinstance(subscopes,str):
        subscopes = [x.strip() for x in subscopes.split(",")]
    else:
        assert(isinstance(subscopes,list))
    return [scope+"/"+x for x in subscopes]

def range_scopes(scope,min,max):
    indexs = range(min,max)
    return [scope%i for i in indexs]

def reshape(tensor,shape,name=None):
    if isinstance(shape,list):
        shape = [ x if (isinstance(x,tf.Tensor) or (x is not None and x >= 0)) else -1 for x in shape]
        return tf.reshape(tensor,shape,name)
    return tf.reshape(tensor,shape,name)

def check_value_in_ckp(sess,scope):
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope)
    logging.info("Check {}".format(scope))
    if len(variables) == 0:
        logging.warning(f"No variables in {scope}.")
        return None
    print(sess.run([tf.reduce_sum(variables[0]),
        tf.reduce_sum(tf.abs(variables[0])),
        tf.reduce_min(variables[0]),
        tf.reduce_max(variables[0]),
        tf.reduce_mean(variables[0])]))

def unstack(value,axis=0,name="unstack",keep_dim=False):
    if keep_dim == False:
        return tf.unstack(value=value,name=name,axis=axis)
    else:
        with tf.name_scope(name):
            res = tf.unstack(value=value,axis=axis)
            res = [tf.expand_dims(x,axis=axis) for x in res]
            return res

def image_zero_mean_unit_range(inputs):
  """Map image values from [0, 255] to [-1, 1]."""
  return (2.0 / 255.0) * tf.to_float(inputs) - 1.0

def mean_pixel(model_variant=None):
  """Gets mean pixel value.

  This function returns different mean pixel value, depending on the input
  model_variant which adopts different preprocessing functions. We currently
  handle the following preprocessing functions:
  (1) _preprocess_subtract_imagenet_mean. We simply return mean pixel value.
  (2) _preprocess_zero_mean_unit_range. We return [127.5, 127.5, 127.5].
  The return values are used in a way that the padded regions after
  pre-processing will contain value 0.

  Args:
    model_variant: Model variant (string) for feature extraction. For
      backwards compatibility, model_variant=None returns _MEAN_RGB.

  Returns:
    Mean pixel value.
  """
  if model_variant is None:
    return _MEAN_RGB
  else:
    return [127.5, 127.5, 127.5]

def num_elements(input):
    if input.get_shape().is_fully_defined():
        shape = input.get_shape().as_list()
        num = 1
        for s in shape:
            num *= s
        return num
    else:
        return tf.reduce_prod(tf.shape(input))
'''
input: [batch_size,D,H,W,C]
size:[ND,NH,NW]
'''
def resize_biliner3d(input,size):
    shape = tf.shape(input)
    input = tf.reshape(input,[shape[0]*shape[1],shape[2],shape[3],shape[4]])
    input = tf.image.resize_bilinear(input, size[1:], align_corners=True)
    shape = [shape[0],shape[1],size[1],size[2],shape[4]]
    input = tf.reshape(input,shape)

    input = tf.transpose(input,perm=[0,3,1,2,4])
    input = tf.reshape(input,[shape[0]*shape[3],shape[1],shape[2],shape[4]])
    input = tf.image.resize_bilinear(input, size[:2], align_corners=True)
    shape = [shape[0],size[0],size[1],size[2],shape[4]]
    input = tf.reshape(input,[shape[0],shape[3],shape[1],shape[2],shape[4]])
    input = tf.transpose(input,perm=[0,2,3,1,4])

    return input

def resize_depth(input,depth):
    shape = tf.shape(input)
    old_type = input.dtype
    input = tf.transpose(input,perm=[0,3,1,2,4])
    input = tf.reshape(input,[shape[0]*shape[3],shape[1],shape[2],shape[4]])
    input = tf.image.resize_bilinear(input, size=(depth,shape[2]), align_corners=True)
    shape = [shape[0],depth,shape[2],shape[3],shape[4]]
    input = tf.reshape(input,[shape[0],shape[3],shape[1],shape[2],shape[4]])
    input = tf.transpose(input,perm=[0,2,3,1,4])

    if old_type != input.dtype:
        input = tf.cast(input,old_type)

    return input


def resize_nearest_neighbor3d(input,size):
    shape = tf.shape(input)
    input = tf.reshape(input,[shape[0]*shape[1],shape[2],shape[3],shape[4]])
    input = tf.image.resize_nearest_neighbor(input, size[1:], align_corners=True)
    shape = [shape[0],shape[1],size[1],size[2],shape[4]]
    input = tf.reshape(input,shape)

    input = tf.transpose(input,perm=[0,3,1,2,4])
    input = tf.reshape(input,[shape[0]*shape[3],shape[1],shape[2],shape[4]])
    input = tf.image.resize_nearest_neighbor(input, size[:2], align_corners=True)
    shape = [shape[0],size[0],size[1],size[2],shape[4]]
    input = tf.reshape(input,[shape[0],shape[3],shape[1],shape[2],shape[4]])
    input = tf.transpose(input,perm=[0,2,3,1,4])

    return input

'''
sparse_indices: [X,Y,...,M,1], 包含了应该设置为sparse_value的index, 格式与top_k返回的格式相同
如[[0]
[1],
[0],
...
]
res:
[X,Y,....,M,dim_size]
'''
def sparse_to_dense(sparse_indices, dim_size, sparse_value, default_value=0):
    old_shape = tf.shape(sparse_indices)
    first_dim_size = tf.reduce_prod(old_shape)
    out_shape = tf.convert_to_tensor([tf.reduce_prod(tf.shape(sparse_indices)),dim_size])
    sparse_indices = tf.reshape(sparse_indices,[-1])
    sparse_indices = tf.stack([tf.range(first_dim_size),sparse_indices],axis=1)
    out_shape = tf.Print(out_shape,[out_shape,tf.shape(sparse_indices)])
    res = tf.sparse_to_dense(sparse_indices,output_shape=out_shape,sparse_values=sparse_value,default_value=default_value)
    res = tf.reshape(res,tf.concat([old_shape[:-1],[dim_size]],axis=0))
    return res


def label_smooth(labels,num_classes,smoothed_value=0.9):
    '''
    :param labels: shape=[batch_size]
    :param num_classes: shape=()
    :param smoothed_value: shape=()
    :return: shape-[batch_size,num_classes]
    '''
    if labels.get_shape().ndims != 1:
        raise ValueError("Labels's should be one dimensional.")
    if not isinstance(num_classes,int):
        raise ValueError("num_classes should be a integer")
    if not isinstance(smoothed_value,float):
        raise ValueError("smoothed_value should be a float")
    default_value = (1.0-smoothed_value)/(num_classes-1)
    res = tf.ones(shape=[tf.shape(labels)[0],num_classes],dtype=tf.float32)*default_value
    res = tf.map_fn(lambda x:set_value(x[0],v=tf.constant([smoothed_value]),index=x[1]),elems=(res,labels),
                    dtype=tf.float32,back_prop=False)

    return res

def label_smoothv1(labels,num_classes,smoothed_value=0.9):
    '''
    :param labels: shape=[batch_size]
    :param num_classes: shape=()
    :param smoothed_value: shape=()
    :return: shape-[batch_size,num_classes]
    '''
    if labels.get_shape().ndims != 1:
        raise ValueError("Labels's should be one dimensional.")
    if not isinstance(num_classes,int):
        raise ValueError("num_classes should be a integer")
    if not isinstance(smoothed_value,float):
        raise ValueError("smoothed_value should be a float")
    default_value = (1.0-smoothed_value)
    def fn(index):
        data = tf.zeros(shape=[num_classes],dtype=tf.float32)
        data0 = set_value(data,v=tf.constant([default_value]),index=tf.constant(0))
        data1 = set_value(data,v=tf.constant([smoothed_value]),index=index)
        return tf.add(data0,data1)
    res_data = tf.map_fn(fn,elems=(labels),
                    dtype=tf.float32,back_prop=False)

    return res_data

def split(datas,num):
    if isinstance(datas,tf.Tensor):
        return tf.split(datas,num_or_size_splits=num)
    else:
        res = []
        for data in datas:
            res.append(tf.split(data,num_or_size_splits=num))
        return res

def probability_case(prob_fn_pairs,scope=None,seed=int(time.time())):
    '''
    :param prob_fn_pairs:[(probs0,fn0),(probs1,fn1),...]
    :param scope:
    :return:
    '''
    with tf.variable_scope(name_or_scope=scope,default_name=f"probability_cond{len(prob_fn_pairs)}"):
        pred_fn_pairs={}
        last_prob = 0.
        p = tf.random_uniform(shape=(),minval=0.,maxval=1.,dtype=tf.float32,seed=seed)
        for pf in prob_fn_pairs:
            fn = pf[1]
            cur_prob = last_prob+pf[0]
            pred = tf.logical_and(tf.greater_equal(p,last_prob),tf.less(p,cur_prob))
            pred_fn_pairs[pred] = fn
            last_prob = cur_prob
        assert math.fabs(last_prob-1.)<1e-2,"Error probabiliby distribultion"

        return tf.case(pred_fn_pairs,exclusive=True)

def fixed_padding(inputs, kernel_size, rate=1):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    rate: An integer, rate for atrous convolution.

  Returns:
    output: A tensor of size [batch, height_out, width_out, channels] with the
      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
  pad_total = kernel_size_effective - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                  [pad_beg, pad_end], [0, 0]])
  return padded_inputs

def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=tf.float32):
  """Creates dense vector with indices set to specific value and rest to zeros.

  This function exists because it is unclear if it is safe to use
    tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
  with indices which are not ordered.
  This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

  Args:
    indices: 1d Tensor with integer indices which are to be set to
        indices_values.
    size: scalar with size (integer) of output Tensor.
    indices_value: values of elements specified by indices in the output vector
    default_value: values of other elements in the output vector.
    dtype: data type.

  Returns:
    dense 1D Tensor of shape [size] with indices set to indices_values and the
        rest set to default_value.
  """
  size = tf.to_int32(size)
  zeros = tf.ones([size], dtype=dtype) * default_value
  values = tf.ones_like(indices, dtype=dtype) * indices_value

  return tf.dynamic_stitch([tf.range(size), tf.to_int32(indices)],
                           [zeros, values])
'''
mask0:[X]
mask1:[Y]
return:
mask:[X]
'''
def merge_mask(mask0,mask1):
    indices = tf.range(tf.reshape(tf.shape(mask0),()))
    indices = tf.boolean_mask(indices,mask0)
    indices = tf.boolean_mask(indices,mask1)
    res = tf.sparse_to_dense(sparse_indices=indices,output_shape=tf.shape(mask0),sparse_values=True,default_value=False)
    return res

def batch_gather(params,indices,name=None):
    if indices.get_shape().ndims <= 2:
        with tf.name_scope(name=name,default_name="batch_gather"):
            return tf.map_fn(lambda x:tf.gather(x[0],x[1]),elems=(params,indices),dtype=params.dtype)
    else:
        return tf.batch_gather(params,indices,name)

def assert_equal(v,values,name=None):
    assert_ops = []
    for i in range(1,len(values)):
        assert_ops.append(tf.assert_equal(values[0],values[i],name=name))
    with tf.control_dependencies(assert_ops):
        return tf.identity(v)

def assert_shape_equal(v,values,name=None):
    shapes = [tf.shape(value) for value in values]
    return assert_equal(v,shapes,name=name)

'''
image:[batch_size,X,H,W,C]
bboxes:[batch_size,X,4] (ymin,xmin,ymax,xmax) in [0,1]
size:(H,W)
output:
[batch_size,box_nr,size[0],size[1],C]
'''
def tf_crop_and_resize(image,bboxes,size):
    img_shape = image.get_shape().as_list()
    batch_size = img_shape[0]
    box_nr = img_shape[1]
    new_img_shape = [img_shape[0]*img_shape[1]]+img_shape[2:]
    bboxes_shape = bboxes.get_shape().as_list()
    new_bboxes_shape = [bboxes_shape[0]*bboxes_shape[1],4]
    image = reshape(image,new_img_shape)
    bboxes = reshape(bboxes,new_bboxes_shape)
    box_ind = tf.range(0,tf.reduce_prod(tf.shape(bboxes)[0]),dtype=tf.int32)
    images = tf.image.crop_and_resize(image,bboxes,box_ind,size)
    shape = images.get_shape().as_list()
    images = reshape(images,[batch_size,box_nr]+shape[1:])
    return images
'''
mask:[N]
output:
indices:[X]
example:
input:[True,True,False,False,False,True]
output:[0,1,5]
'''
def mask_to_indices(mask):
    indices = tf.range(tf.reshape(tf.shape(mask),()),dtype=tf.int32)
    return tf.boolean_mask(indices,mask)

def indices_to_mask(indices,size):
    mask = tf.cast(indices_to_dense_vector(indices,size,1,default_value=0,dtype=tf.int32),tf.bool)
    _,ind = tf.nn.top_k(indices,tf.reshape(tf.shape(indices),()))
    ind = tf.reverse(ind,axis=[0])
    return mask,ind

def batch_indices_to_mask(indices,lens,size):
    if indices.get_shape().is_fully_defined():
        ind_size = indices.get_shape().as_list()[1]
    else:
        ind_size = tf.shape(indices)[1]
    def fn(ind,l):
        ind = ind[:l]
        mask,ind = indices_to_mask(ind,size)
        ind = tf.pad(ind,tf.convert_to_tensor([[0,ind_size-l]]))
        return mask,ind
    return tf.map_fn(lambda x:fn(x[0],x[1]),elems=(indices,lens),dtype=(tf.bool,tf.int32),back_prop=False)

def batch_boolean_mask(data,mask,size):
    if not isinstance(data,tf.Tensor):
        data = tf.convert_to_tensor(data)
    def fn(d,m):
        d = tf.boolean_mask(d,m)
        d = tf.pad(d,[[0,size-tf.shape(d)[0]]]+[0,0]*(d.get_shape().ndims-1))
        return d
    return tf.map_fn(lambda x:fn(x[0],x[1]),elems=(data,mask),dtype=(data.dtype),back_prop=False)

def Print(data,*inputs,**kwargs):
    op = tf.print(*inputs,**kwargs)
    with tf.control_dependencies([op]):
        return tf.identity(data)

'''
indicator:[X],tf.bool
'''
def subsample_indicator(indicator, num_samples):
    indices = tf.where(indicator)
    indices = tf.random_shuffle(indices)
    indices = tf.reshape(indices, [-1])
    if isinstance(num_samples,tf.Tensor) and num_samples.dtype != tf.int32:
        num_samples = tf.cast(num_samples,tf.int32)
    num_samples = tf.minimum(tf.size(indices), num_samples)
    selected_indices = tf.slice(indices, [0], tf.reshape(num_samples, [1]))

    selected_indicator = indices_to_dense_vector(selected_indices,
                                                     tf.shape(indicator)[0])

    return tf.equal(selected_indicator, 1)


def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape

def nearest_neighbor_upsampling(input_tensor, scale=None, height_scale=None,
                                width_scale=None,scope=None):
  """Nearest neighbor upsampling implementation.

  Nearest neighbor upsampling function that maps input tensor with shape
  [batch_size, height, width, channels] to [batch_size, height * scale
  , width * scale, channels]. This implementation only uses reshape and
  broadcasting to make it TPU compatible.

  Args:
    input_tensor: A float32 tensor of size [batch, height_in, width_in,
      channels].
    scale: An integer multiple to scale resolution of input data in both height
      and width dimensions.
    height_scale: An integer multiple to scale the height of input image. This
      option when provided overrides `scale` option.
    width_scale: An integer multiple to scale the width of input image. This
      option when provided overrides `scale` option.
  Returns:
    data_up: A float32 tensor of size
      [batch, height_in*scale, width_in*scale, channels].

  Raises:
    ValueError: If both scale and height_scale or if both scale and width_scale
      are None.
  """
  if not scale and (height_scale is None or width_scale is None):
    raise ValueError('Provide either `scale` or `height_scale` and'
                     ' `width_scale`.')
  with tf.name_scope(scope,'nearest_neighbor_upsampling'):
    h_scale = scale if height_scale is None else height_scale
    w_scale = scale if width_scale is None else width_scale
    (batch_size, height, width,
     channels) = combined_static_and_dynamic_shape(input_tensor)
    output_tensor = tf.reshape(
        input_tensor, [batch_size, height, 1, width, 1, channels]) * tf.ones(
            [1, 1, h_scale, 1, w_scale, 1], dtype=input_tensor.dtype)
    return tf.reshape(output_tensor,
                      [batch_size, height * h_scale, width * w_scale, channels])

def nearest_neighbor_downsampling(input_tensor, scale=None, height_scale=None,
                                width_scale=None):
    if not scale and (height_scale is None or width_scale is None):
        raise ValueError('Provide either `scale` or `height_scale` and'
                         ' `width_scale`.')
    with tf.name_scope('nearest_neighbor_downsampling'):
        h_scale = scale if height_scale is None else height_scale
        w_scale = scale if width_scale is None else width_scale
        (batch_size, height, width,
         channels) = combined_static_and_dynamic_shape(input_tensor)
        output_tensor = tf.reshape(
            input_tensor, [batch_size, height//h_scale, h_scale, width//w_scale, w_scale, channels])
        return output_tensor[:,:,0,:,0,:]
def channel_upsample(input_tensor,scale=None,height_scale=None,width_scale=None):
    if not scale and (height_scale is None or width_scale is None):
        raise ValueError('Provide either `scale` or `height_scale` and'
                         ' `width_scale`.')
    with tf.name_scope('channel_upsampling'):
        h_scale = scale if height_scale is None else height_scale
        w_scale = scale if width_scale is None else width_scale
        (batch_size, height, width,
         channels) = combined_static_and_dynamic_shape(input_tensor)
        out_channels = channels//(h_scale*w_scale)
        output_tensor = tf.reshape(
            input_tensor, [batch_size, height, width, h_scale,w_scale,out_channels])
        output_tensor = tf.transpose(output_tensor,[0,1,3,2,4,5])
        output_tensor = tf.reshape(output_tensor,[batch_size,height*h_scale,width*w_scale,out_channels])
        return output_tensor

def show_return_shape(func,message=None):
    @wraps(func)
    def wraps_func(*args,**kwargs):
        res = func(*args,**kwargs)
        if isinstance(res,dict):
            datas = []
            key = None
            for k, d in res.items():
                if not isinstance(d, tf.Tensor):
                    datas.append(tf.constant("N.A", dtype=tf.string))
                else:
                    datas.append(tf.shape(d))
                    if key is None:
                        key = k
            res[key] = tf.Print(res[key], datas, summarize=100,message=message)
        elif not isinstance(res,Iterable):
            res = tf.Print(res,[tf.shape(res)],summarize=100)
        else:
            datas = []
            index = -1
            for i,d in enumerate(res):
                if not isinstance(d,tf.Tensor):
                    datas.append(tf.constant("N.A",dtype=tf.string))
                else:
                    datas.append(tf.shape(d))
                    if index<0:
                        index = i
            res = list(res)
            res[index] = tf.Print(res[index],datas,summarize=100,message=message)
        return res
    return wraps_func

if __name__ == "__main__":
    wmlu.show_list(get_variables_in_ckpt_in_dir("../../mldata/faster_rcnn_resnet101/"))

