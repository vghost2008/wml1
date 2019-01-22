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
import img_utils as wmli
import time

_HASH_TABLE_COLLECTION = "HASH_TABLE"
_MEAN_RGB = [123.15, 115.90, 103.06]


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


def isSingleValueTensor(var):
    if not var.get_shape().is_fully_defined():
        return False
    dim = 1
    shape = var.get_shape().as_list()
    for v in shape:
        dim *= v
    return dim==1


def show_values(values,name=None):
    if name is not None:
        print(name)
    for v in values:
        print(v)

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

def random_saturation(image,gray_image=None,minval=0.0,maxval=1.0,scope=None):
    with tf.name_scope(scope, 'random_saturation', [image]):
        if gray_image is None:
            gray_image = wmli.rgb_to_grayscale(image,keep_channels=True)
        ratio = tf.random_uniform(shape=(),
                                  minval=minval,
                                  maxval=maxval,
                                  dtype=tf.float32,
                                  seed=int(time.time()))
        return gray_image*ratio + image*(1.0-ratio)


def distort_color(image, color_ordering=0, fast_mode=False, scope=None):

    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            b_max_delta = 32./255
            c_lower = 0.5
            c_upper = 1.5
            s_lower = 0.5
            s_upper = 1.5
            h_max_delta = 0.2
            if color_ordering == 0:
                image = tf.image.random_brightness(image, b_max_delta)
                image = tf.image.random_contrast(image, lower=c_lower, upper=c_upper)
            elif color_ordering == 1:
                image = tf.image.random_contrast(image, lower=c_lower, upper=c_upper)
                image = tf.image.random_brightness(image, b_max_delta)
            elif color_ordering == 2:
                image = tf.image.random_brightness(image, b_max_delta)
                image = tf.image.random_contrast(image, lower=c_lower, upper=c_upper)
                image = tf.image.random_saturation(image, lower=s_lower, upper=s_upper)
            elif color_ordering == 3:
                image = tf.image.random_saturation(image, lower=s_lower, upper=s_upper)
                image = tf.image.random_brightness(image, b_max_delta)
                image = tf.image.random_contrast(image, lower=c_lower, upper=c_upper)
            elif color_ordering == 4:
                image = tf.image.random_saturation(image, lower=s_lower, upper=s_upper)
                image = tf.image.random_brightness(image, b_max_delta)
                image = tf.image.random_contrast(image, lower=c_lower, upper=c_upper)
                image = tf.image.random_hue(image, max_delta=h_max_delta)
            elif color_ordering == 5:
                image = tf.image.random_hue(image, max_delta=h_max_delta)
                image = tf.image.random_saturation(image, lower=s_lower, upper=s_upper)
                image = tf.image.random_brightness(image, b_max_delta)
                image = tf.image.random_contrast(image, lower=c_lower, upper=c_upper)
            elif color_ordering == 6:
                image = tf.image.random_saturation(image, lower=s_lower, upper=s_upper)
                image = tf.image.random_hue(image, max_delta=h_max_delta)
                image = tf.image.random_brightness(image, b_max_delta)
                image = tf.image.random_contrast(image, lower=c_lower, upper=c_upper)
            elif color_ordering == 7:
                return image
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        return tf.clip_by_value(image, 0.0, 1.0)


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

def variable_summaries(var,name):
    if var.dtype != tf.float32:
        var = tf.cast(var, tf.float32)
    mean = tf.reduce_mean(var)
    tf.summary.scalar(name+'/max', tf.reduce_max(var))
    tf.summary.scalar(name+'/min', tf.reduce_min(var))
    tf.summary.scalar(name+'/mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(name+'/stddev', stddev)
    tf.summary.histogram(name+'/hisogram', var)

def variable_summaries_v2(var, name):
    if var.dtype != tf.float32:
        var = tf.cast(var, tf.float32)
    if isSingleValueTensor(var):
        var = tf.reshape(var,())
        tf.summary.scalar(name, var)
        return
    mean = tf.reduce_mean(var)
    tf.summary.scalar(name+'/max', tf.reduce_max(var))
    tf.summary.scalar(name+'/min', tf.reduce_min(var))
    tf.summary.scalar(name+'/mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(name+'/stddev', stddev)
    tf.summary.histogram(name+'/hisogram', var)

def image_summaries(var,name,max_outputs=3):
    if var.get_shape().ndims==3:
        var = tf.expand_dims(var,dim=0)
    tf.summary.image(name,var,max_outputs=max_outputs)

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
    handle with the situation which x or index's shape is not fully defined.
    :param x: (Y,M,N,...)
    :param index: (Y)
    :return: (Y,N,...)
    '''
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
'''
对图像image进行剪切，生成四个角及中间五个不同位置的图像
如果resize_size不为None， 那么生成的图像会被缩放为resize_size指定的大小
'''
def crop_image(image,width,height,resize_size=None):
    shape = tf.shape(image)
    images = []
    img = tf.image.crop_to_bounding_box(image,0,0,height,width)
    images.append(img)
    img = tf.image.crop_to_bounding_box(image,shape[0]-height,shape[1]-width,height,width)
    images.append(img)
    img = tf.image.crop_to_bounding_box(image, (shape[0] - height)/2, (shape[1] - width)/2, height, width)
    images.append(img)
    img = tf.image.crop_to_bounding_box(image, 0, shape[1] - width, height, width)
    images.append(img)
    img = tf.image.crop_to_bounding_box(image, shape[0] - height, 0, height, width)
    images.append(img)

    if resize_size is not None:
        images = tf.stack(images,axis=0)
        return tf.image.resize_images(images,resize_size)
    else:
        return tf.stack(images,axis=0)

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
        subscopes = [ x.strip() for x in subscopes.split(",")]
    else:
        assert(isinstance(subscopes,list))
    return [scope+"/"+x for x in subscopes]

def range_scopes(scope,min,max):
    indexs = range(min,max)
    return [scope%i for i in indexs]

def reshape(tensor,shape,name=None):
    if isinstance(shape,list):
        shape = [ x if (x is not None and x >= 0) else -1 for x in shape]
        return tf.reshape(tensor,shape,name)
    return tf.reshape(tensor,shape,name)

def check_value_in_ckp(sess,scope):
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope)
    print("Check {}".format(scope))
    if len(variables) == 0:
        print(f"No variables in {scope}.")
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

def split(datas,num):
    if isinstance(datas,tf.Tensor):
        return tf.split(datas,num_or_size_splits=num)
    else:
        res = []
        for data in datas:
            res.append(tf.split(data,num_or_size_splits=num))
        return res

def probability_case(prob_fn_pairs,scope=None):
    '''
    :param prob_fn_pairs:[(probs0,fn0),(probs1,fn1),...]
    :param scope:
    :return:
    '''
    with tf.variable_scope(name_or_scope=scope,default_name="probability_cond"):
        pred_fn_pairs={}
        last_prob = 0.
        p = tf.random_uniform(shape=(),minval=0.,maxval=1.,dtype=tf.float32)
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

if __name__ == "__main__":
    wmlu.show_list(get_variables_in_ckpt_in_dir("../../mldata/faster_rcnn_resnet101/"))
