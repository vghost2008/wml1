#coding=utf-8
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python import pywrap_tensorflow
import wml_utils as wmlu
import os
import numpy as np
import logging
import wsummary
import basic_tftools as btf
from wtfop.wtfop_ops import set_value


_HASH_TABLE_COLLECTION = "HASH_TABLE"
_MEAN_RGB = [123.15, 115.90, 103.06]

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
combined_static_and_dynamic_shape = btf.combined_static_and_dynamic_shape
batch_gather = btf.batch_gather
show_return_type = btf.show_return_shape
add_name_scope = btf.add_name_scope
add_variable_scope = btf.add_variable_scope
probability_case = btf.probability_case
indices_to_dense_vector = btf.indices_to_dense_vector

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

def top_k_mask_nd(value,k=1):
    assert value.shape.ndims>1, "error dim size"
    shape = btf.combined_static_and_dynamic_shape(value)
    N = 1
    for i in range(len(shape)-1):
        N = N*shape[i]
    value = tf.reshape(value,[N,shape[-1]])
    values, indics = tf.nn.top_k(value, k)
    indics, _ = tf.nn.top_k(-indics, k)
    indics1 = -indics
    indics0 = tf.reshape(tf.range(N),[N,1])
    indics0 = tf.tile(indics0,[1,k])
    indics = tf.reshape(indics1,[-1,1])
    indics0 = tf.reshape(indics0,[-1,1])
    indics = tf.concat([indics0,indics],axis=1)
    res = tf.cast(tf.sparse_to_dense(indics,[N,shape[-1]], 1), tf.bool)
    res = tf.reshape(res,shape)
    indics1 = tf.reshape(indics1,shape[:-1]+[k])
    return res,indics1


def top_k_mask_1d(value,k=1):
    assert value.shape.ndims==1, "error dim size"
    values, indics = tf.nn.top_k(value, k)
    indics, _ = tf.nn.top_k(-indics, k)
    indics = -indics
    res = tf.cast(tf.sparse_to_dense(indics, value.shape, 1), tf.bool)
    return res,indics

def top_k_mask(value,k=1,shape=None,return_indices=False):
    with tf.name_scope("top_k_mask"):
        if value.shape.ndims == 1:
            res,indices = top_k_mask_1d(value,k=k)
        else:
            res,indices = top_k_mask_nd(value,k=k)
        if shape is not None:
            res = tf.reshape(res,shape)
        if return_indices:
            return res,indices
        else:
            return res

def random_top_k_mask_nd(value,k=3,nr=1):
    assert value.shape.ndims>1, "error dim size"
    shape = btf.combined_static_and_dynamic_shape(value)
    N = 1
    for i in range(len(shape)-1):
        N = N*shape[i]
    value = tf.reshape(value,[N,shape[-1]])
    values, indics = tf.nn.top_k(value, k)
    indics = tf.transpose(indics)
    indics = tf.random_shuffle(indics)
    indics = indics[:nr,:]
    indics = tf.transpose(indics)
    indics, _ = tf.nn.top_k(-indics, nr)
    indics1 = -indics
    indics0 = tf.reshape(tf.range(N),[N,1])
    indics0 = tf.tile(indics0,[1,nr])
    indics = tf.reshape(indics1,[-1,1])
    indics0 = tf.reshape(indics0,[-1,1])
    indics = tf.concat([indics0,indics],axis=1)
    res = tf.cast(tf.sparse_to_dense(indics,[N,shape[-1]], 1), tf.bool)
    res = tf.reshape(res,shape)
    return res,indics1


def random_top_k_mask_1d(value,k=3,nr=1):
    assert value.shape.ndims==1, "error dim size"
    values, indics = tf.nn.top_k(value, k)
    indics = tf.random_shuffle(indics)
    indics = indics[:nr]
    indics, _ = tf.nn.top_k(-indics, nr)
    indics = -indics
    res = tf.cast(tf.sparse_to_dense(indics, value.shape, 1), tf.bool)
    return res,indics

'''
从value中选出得分最高的k个，再从k个中随机选nr个返回
'''
def random_top_k_mask(value,k=3,nr=1,shape=None,return_indices=False):
    with tf.name_scope("top_k_mask"):
        if value.shape.ndims == 1:
            res,indices = random_top_k_mask_1d(value,k=k,nr=nr)
        else:
            res,indices = random_top_k_mask_nd(value,k=k,nr=nr)
        if shape is not None:
            res = tf.reshape(res,shape)
        if return_indices:
            return res,indices
        else:
            return res

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

def check_value_in_ckpv2(sess,variable):
    graph = tf.get_default_graph()
    variable = graph.get_tensor_by_name(variable)
    print(sess.run([tf.reduce_sum(variable),
                    tf.reduce_sum(tf.abs(variable)),
                    tf.reduce_min(variable),
                    tf.reduce_max(variable),
                    tf.reduce_mean(variable)]))

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
        return input.get_shape().num_elements()
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
image:[batch_size,H,W,C]/[H,W,C]
bboxes:[batch_size,X,4]/[X,4] (ymin,xmin,ymax,xmax) in [0,1]
length::[batch_size]
size:(H,W)
output:
[box_nr,size[0],size[1],C]
'''
def tf_crop_and_resizev2(image,bboxes,size,lengths=None):
    if len(image.get_shape())==3:
        assert len(bboxes.get_shape())==2, "error box shape"
        image = tf.expand_dims(image,axis=0)
        bboxes = tf.expand_dims(bboxes,axis=0)
    assert len(image.get_shape())==4,"error image shape"
    assert len(bboxes.get_shape())==3,"error bboxes shape"
    B,H,W,C = btf.combined_static_and_dynamic_shape(image)
    _,X,_ = btf.combined_static_and_dynamic_shape(bboxes)
    bboxes = tf.reshape(bboxes,[B*X,4])

    index = tf.range(B)
    index = tf.tile(tf.reshape(index,[B,1]),[1,X])
    index = tf.reshape(index,[-1])
    if lengths is not None:
        mask = tf.sequence_mask(lengths,maxlen=X)
        mask = tf.reshape(mask,[-1])
        bboxes = tf.boolean_mask(bboxes,mask)
        index = tf.boolean_mask(index,mask)
    images = tf.image.crop_and_resize(image,bboxes,index,size)
    return images
'''
image:[batch_size,X,H,W,C]/[X,H,W,C]
bboxes:[batch_size,X,4]/[X,4] (ymin,xmin,ymax,xmax) in [0,1]
size:(H,W)
lengths: [batch_size]
output:
[batch_size,box_nr,size[0],size[1],C]/ [box_nr,size[0],size[1],C]
or [Y,size[0],size[1],C] if lengths is not None
'''
def tf_crop_and_resize(image,bboxes,size,lengths=None):
    if len(image.get_shape()) == 4:
        image = tf.expand_dims(image,axis=0)
        bboxes = tf.expand_dims(bboxes,axis=0)
        return tf.squeeze(batch_tf_crop_and_resize(image,bboxes,size),axis=0)
    elif len(image.get_shape()) == 5:
        res = batch_tf_crop_and_resize(image,bboxes,size)
        if lengths is not None:
            B,X,H,W,C = btf.combined_static_and_dynamic_shape(image)
            mask = tf.reshape(tf.sequence_mask(lengths,X),[-1])
            res = tf.reshape(res,[-1,size[0],size[1],C])
            res = tf.boolean_mask(res,mask)
        return res
    else:
        raise Exception("Error image ndims.")
'''
image:[batch_size,X,H,W,C]
bboxes:[batch_size,X,4] (ymin,xmin,ymax,xmax) in [0,1]
size:(H,W)
output:
[batch_size,box_nr,size[0],size[1],C]
'''
def batch_tf_crop_and_resize(image,bboxes,size):
    img_shape = btf.combined_static_and_dynamic_shape(image)
    batch_size = img_shape[0]
    box_nr = img_shape[1]
    new_img_shape = [img_shape[0]*img_shape[1]]+img_shape[2:]
    bboxes_shape = btf.combined_static_and_dynamic_shape(bboxes)
    new_bboxes_shape = [bboxes_shape[0]*bboxes_shape[1],4]
    image = reshape(image,new_img_shape)
    bboxes = reshape(bboxes,new_bboxes_shape)
    box_ind = tf.range(0,tf.reduce_prod(tf.shape(bboxes)[0]),dtype=tf.int32)
    images = tf.image.crop_and_resize(image,bboxes,box_ind,size)
    shape = btf.combined_static_and_dynamic_shape(images)
    images = reshape(images,[batch_size,box_nr]+shape[1:])
    return images

mask_to_indices = btf.mask_to_indices

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

'''
每一个element分别执行boolean mask并pad到size大小
data:[N,X]
mask:[N,X]
size:()
return:
[N,size]
'''
def batch_boolean_mask(data,mask,size):
    if not isinstance(data,tf.Tensor):
        data = tf.convert_to_tensor(data)
    def fn(d,m):
        d = tf.boolean_mask(d,m)
        padding = [[0,size-tf.shape(d)[0]]]
        if d.get_shape().ndims>1:
            padding = padding+[[0,0]]*(d.get_shape().ndims-1)
        d = tf.pad(d,padding)
        return d
    return tf.map_fn(lambda x:fn(x[0],x[1]),elems=(data,mask),dtype=(data.dtype),back_prop=False)

'''
每一个element分别执行boolean mask并 concat在一起
data:[N,X]
mask:[N,X]
N must be full defined
return:
[Y]
'''
def batch_boolean_maskv2(data,mask):
    if not isinstance(data,tf.Tensor):
        data = tf.convert_to_tensor(data)
    res = []
    shape = btf.combined_static_and_dynamic_shape(data)
    for i in range(shape[0]):
        res.append(tf.boolean_mask(data[i],mask[i]))
    return tf.concat(res,axis=0)

'''
每一个element先用indices gather再执行boolean mask并 concat在一起
data:[M,X]
indices:[N,X]
mask:[N,X]
N must be full defined
return:
[Y]
'''
def batch_boolean_maskv3(data,indices,mask):
    if not isinstance(data,tf.Tensor):
        data = tf.convert_to_tensor(data)
    res = []
    shape = btf.combined_static_and_dynamic_shape(data)
    for i in range(shape[0]):
        indx = tf.boolean_mask(indices[i],mask[i])
        d = tf.gather(data[i],indx)
        res.append(d)
    return tf.concat(res,axis=0)

def Print(data,*inputs,**kwargs):
    op = tf.print(*inputs,**kwargs)
    with tf.control_dependencies([op]):
        return tf.identity(data)

def print_tensor_shape(input_,data,name=None,summarize=100):
    data = [tf.shape(x) for x in data]
    if name is not None:
        data = [tf.constant(name+": ")]+data
    return tf.Print(input_,data,summarize=summarize)

'''
indicator:[X],tf.bool
return:
[x]:tf.bool
'''
def subsample_indicator(indicator, num_samples):
    with tf.name_scope("sample_indicator"):
        indicator_shape = btf.combined_static_and_dynamic_shape(indicator)
        indices = tf.where(indicator)
        indices = tf.random_shuffle(indices)
        indices = tf.reshape(indices, [-1])
        if isinstance(num_samples,tf.Tensor) and num_samples.dtype != tf.int32:
            num_samples = tf.cast(num_samples,tf.int32)
        num_samples = tf.minimum(tf.size(indices), num_samples)
        selected_indices = tf.slice(indices, [0], tf.reshape(num_samples, [1]))

        selected_indicator = indices_to_dense_vector(selected_indices,
                                                         indicator_shape[0])
        selected_indicator = tf.reshape(selected_indicator,indicator_shape)

        return tf.equal(selected_indicator, 1)

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

def select_image_by_mask(image,mask):
    '''

    :param image: [batch_size,H,W,C]
    :param mask: [batch_size,N], tf.bool
    :return:
    [X,H,W,C], X = tf.reduce_sum(mask)
    '''
    with tf.name_scope("select_image_by_mask"):
        batch_size,H,W,C = combined_static_and_dynamic_shape(image)
        index = tf.reshape(tf.range(batch_size),[batch_size,1])*tf.ones_like(mask,dtype=tf.int32)
        index = tf.boolean_mask(index,mask)
        return tf.gather(image,index)
def sort_data(key,datas):
    size = tf.shape(key)[0]
    values,indices = tf.nn.top_k(key,k=size)
    datas = [tf.gather(x,indices) for x in datas]
    return [values,indices],datas

def get_pad_shapes_for_padded_batch(dataset):
    shapes = dataset.output_shapes
    res = {}
    for k,v in shapes.items():
        shape = v.as_list()
        res[k] = shape
    return res

if __name__ == "__main__":
    wmlu.show_list(get_variables_in_ckpt_in_dir("../../mldata/faster_rcnn_resnet101/"))

