#coding=utf-8
import tensorflow as tf
from functools import wraps,partial
from collections import Iterable
import time
import math
from collections import OrderedDict

def static_or_dynamic_map_fn(fn, elems, dtype=None,
                             parallel_iterations=32, back_prop=True):
  """Runs map_fn as a (static) for loop when possible.

  This function rewrites the map_fn as an explicit unstack input -> for loop
  over function calls -> stack result combination.  This allows our graphs to
  be acyclic when the batch size is static.
  For comparison, see https://www.tensorflow.org/api_docs/python/tf/map_fn.

  Note that `static_or_dynamic_map_fn` currently is not *fully* interchangeable
  with the default tf.map_fn function as it does not accept nested inputs (only
  Tensors or lists of Tensors).  Likewise, the output of `fn` can only be a
  Tensor or list of Tensors.

  TODO(jonathanhuang): make this function fully interchangeable with tf.map_fn.

  Args:
    fn: The callable to be performed. It accepts one argument, which will have
      the same structure as elems. Its output must have the
      same structure as elems.
    elems: A tensor or list of tensors, each of which will
      be unpacked along their first dimension. The sequence of the
      resulting slices will be applied to fn.
    dtype:  (optional) The output type(s) of fn. If fn returns a structure of
      Tensors differing from the structure of elems, then dtype is not optional
      and must have the same structure as the output of fn.
    parallel_iterations: (optional) number of batch items to process in
      parallel.  This flag is only used if the native tf.map_fn is used
      and defaults to 32 instead of 10 (unlike the standard tf.map_fn default).
    back_prop: (optional) True enables support for back propagation.
      This flag is only used if the native tf.map_fn is used.

  Returns:
    A tensor or sequence of tensors. Each tensor packs the
    results of applying fn to tensors unpacked from elems along the first
    dimension, from first to last.
  Raises:
    ValueError: if `elems` a Tensor or a list of Tensors.
    ValueError: if `fn` does not return a Tensor or list of Tensors
  """
  if isinstance(elems, list):
    for elem in elems:
      if not isinstance(elem, tf.Tensor):
        raise ValueError('`elems` must be a Tensor or list of Tensors.')

    elem_shapes = [elem.shape.as_list() for elem in elems]
    # Fall back on tf.map_fn if shapes of each entry of `elems` are None or fail
    # to all be the same size along the batch dimension.
    for elem_shape in elem_shapes:
      if (not elem_shape or not elem_shape[0]
          or elem_shape[0] != elem_shapes[0][0]):
        return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
    arg_tuples = zip(*[tf.unstack(elem) for elem in elems])
    outputs = [fn(arg_tuple) for arg_tuple in arg_tuples]
  else:
    if not isinstance(elems, tf.Tensor):
      raise ValueError('`elems` must be a Tensor or list of Tensors.')
    elems_shape = elems.shape.as_list()
    if not elems_shape or not elems_shape[0]:
      return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
    outputs = [fn(arg) for arg in tf.unstack(elems)]
  # Stack `outputs`, which is a list of Tensors or list of lists of Tensors
  if all([isinstance(output, tf.Tensor) for output in outputs]):
    return tf.stack(outputs)
  else:
    if all([isinstance(output, list) for output in outputs]):
      if all([all(
          [isinstance(entry, tf.Tensor) for entry in output_list])
              for output_list in outputs]):
        return [tf.stack(output_tuple) for output_tuple in zip(*outputs)]
  raise ValueError('`fn` should return a Tensor or a list of Tensors.')

def isSingleValueTensor(var):
  if not var.get_shape().is_fully_defined():
    return False
  dim = 1
  shape = var.get_shape().as_list()
  for v in shape:
    dim *= v
  return dim == 1

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

'''
mask:[B,N]
output:
indices:[B,N]
lengths:[B]
'''
def batch_mask_to_indices(mask):
    with tf.name_scope("batch_mask_to_indices"):
        batch_size,N = combined_static_and_dynamic_shape(mask)
        def fn(mask):
            indices = mask_to_indices(mask)
            len = tf.shape(indices)[0]
            indices = tf.pad(indices,paddings=[[0,N-len]])
            return indices,len
        indices,lens = tf.map_fn(fn,elems=mask,dtype=(tf.int32,tf.int32),back_prop=False)

    return indices,lens

'''
params:[batch_size,X,...]
indices:[batch_size,...]
如果indices:[batch_size]那么返回[batch_size,...]
'''
def batch_gather(params,indices,name=None):
    if indices.get_shape().ndims <= 1:
        with tf.name_scope(name=name, default_name="batch_gather"):
            indices_shape = combined_static_and_dynamic_shape(indices)
            batch_indices = tf.range(indices_shape[0])
            indices = tf.reshape(indices,[-1])
            indices = tf.stack([batch_indices,indices],axis=1)
            return tf.gather_nd(params,indices)
    elif indices.get_shape().ndims <= 2:
        with tf.name_scope(name=name,default_name="batch_gather"):
            return tf.map_fn(lambda x:tf.gather(x[0],x[1]),elems=(params,indices),dtype=params.dtype)
    else:
        return tf.batch_gather(params,indices,name)

def show_input_shape(func,message=None):
    @wraps(func)
    def wraps_func(*args,**kwargs):
        data = []
        for d in args:
            data.append(d)
        for k,v in kwargs:
            data.append(v)
        datas = []
        index = -1
        for i,d in enumerate(data):
            if not isinstance(d,tf.Tensor):
                datas.append(tf.constant("N.A",dtype=tf.string))
            else:
                datas.append(tf.shape(d))
                if index<0:
                    index = i
        res = list(data)
        res[index] = tf.Print(res[index],datas,summarize=100,message=message)
        res = func(*args,**kwargs)
        return res
    return wraps_func

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

def add_name_scope(func):
    def wraps_func(*args,**kwargs):
        with tf.name_scope(func.__name__):
            return func(*args,**kwargs)
    return wraps_func

def add_variable_scope(func):
    def wraps_func(*args,**kwargs):
        with tf.variable_scope(func.__name__):
            return func(*args,**kwargs)
    return wraps_func

def probability_case(prob_fn_pairs,scope=None,seed=int(time.time()),prob=None):
    '''
    :param prob_fn_pairs:[(probs0,fn0),(probs1,fn1),...]
    :param scope:
    :return:
    '''
    with tf.variable_scope(name_or_scope=scope,default_name=f"probability_cond{len(prob_fn_pairs)}"):
        pred_fn_pairs=OrderedDict()
        last_prob = 0.
        if prob is None:
            p = tf.random_uniform(shape=(),minval=0.,maxval=1.,dtype=tf.float32,seed=seed)
        else:
            p = prob
        for pf in prob_fn_pairs:
            fn = pf[1]
            cur_prob = last_prob+pf[0]
            pred = tf.logical_and(tf.greater_equal(p,last_prob),tf.less(p,cur_prob))
            pred_fn_pairs[pred] = fn
            last_prob = cur_prob
        assert math.fabs(last_prob-1.)<1e-2,"Error probabiliby distribultion"

        return tf.case(pred_fn_pairs,exclusive=True)
def _identity(x):
    return x
def select_in_list(datas,index,scope=None):
    with tf.name_scope(scope,default_name=f"select_in_list{len(datas)}"):
        pred_fn_pairs = OrderedDict()
        for i,d in enumerate(datas):
            pred_fn_pairs[tf.equal(index,i)] = partial(_identity,d)
        return tf.case(pred_fn_pairs,exclusive=True)

@add_name_scope
def twod_indexs_to_oned_indexs(indexs,depth=None):
    '''
    :param indexs: [N,M]
    :param depth: the offset
    :return: [N*M],
    res[0:M] = indexs[0]
    res[M,M*2] = indexs[1]+depth
    ...
    '''
    N,M = combined_static_and_dynamic_shape(indexs)

    if depth is None:
        depth =  M

    offset = tf.reshape(tf.range(N,dtype=indexs.dtype)*depth,[N,1])*tf.ones([N,M],dtype=indexs.dtype)
    offset = tf.reshape(offset,[-1])
    indexs = tf.reshape(indexs,[-1])
    return indexs+offset

def safe_reduce_mean(input_tensor,
                axis=None,
                keepdims=None,
                name=None,
                reduction_indices=None,
                default_value=0,
                keep_dims=None):
    nr = tf.reduce_prod(tf.shape(input_tensor))
    shape = combined_static_and_dynamic_shape(input_tensor)
    def get_default_value():
        if axis is not None:
            if axis<len(shape)-1:
                shape_i = shape[:axis]+shape[axis+1:]
            else:
                shape_i = shape[:axis]
        else:
            shape_i = ()

        if math.fabs(default_value)<1e-8:
            return tf.zeros(shape_i,dtype=input_tensor.dtype)

    def get_normal_value():
        return tf.reduce_mean(input_tensor,axis=axis,keepdims=keepdims,name=name,
                              reduction_indices=reduction_indices,
                              keep_dims=keep_dims)
    return tf.cond(tf.greater(nr,0),get_normal_value,get_default_value)

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

def channel(x,format="NHWC"):
    if format == "NHWC":
        return x.get_shape().as_list()[-1]
    elif format == "NCHW":
        return x.get_shape().as_list()[1]
    else:
        raise NotImplementedError("Error")

def batch_size(x):
    return combined_static_and_dynamic_shape(x)[0]

def expand_dim_to(input,axis,repeat=1,name=None):
    with tf.name_scope(name,default_name=f"expand_dim_to{repeat}"):
        input = tf.expand_dims(input,axis=axis)
        assert repeat>0,f"Error repeat {repeat}"
        if repeat != 1:
            multiples = [1]*len(input.get_shape())
            multiples[axis] = repeat
            return tf.tile(input,multiples=multiples)
        else:
            return input
