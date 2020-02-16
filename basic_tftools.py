#coding=utf-8
import tensorflow as tf
from functools import wraps
from collections import Iterable

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
    if indices.get_shape().ndims <= 2:
        with tf.name_scope(name=name,default_name="batch_gather"):
            return tf.map_fn(lambda x:tf.gather(x[0],x[1]),elems=(params,indices),dtype=params.dtype)
    else:
        return tf.batch_gather(params,indices,name)

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