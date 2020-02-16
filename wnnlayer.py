import tensorflow as tf
import wtfop.wtfop_ops as wtfop
import wml_tfutils as wmlt
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
import nlp.wlayers as nlpl
import numpy as np
import time
from tensorflow.contrib.framework.python.ops import add_arg_scope

DATA_FORMAT_NHWC = 'NHWC'
slim = tf.contrib.slim

class ConvGRUCell(tf.nn.rnn_cell.RNNCell):
    '''
    shape: [H,W] net spatial shape
    filters: filter size, which is the output channel num, e.g 128
    kernel: kernel size, e.g [3,3]
    '''
    def __init__(self, shape, filters, kernel, activation=tf.tanh,
                 reuse=None):
        super(ConvGRUCell, self).__init__(_reuse=reuse)
        self._filters = filters
        self._kernel = kernel
        self._activation = activation
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None

    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    def call(self, x, h):
        #x's channel size
        channels = x.shape[self._feature_axis].value

        with tf.variable_scope('gates'):
            inputs = tf.concat([x, h], axis=self._feature_axis)
            n = channels + self._filters
            #at lest output 2 channels
            m = 2 * self._filters if self._filters > 1 else 2
            #W's shape is spatial_filter_shape + [in_channels, out_channels]
            W = tf.get_variable('kernel', self._kernel + [n, m])
            y = tf.nn.convolution(inputs, W, 'SAME', data_format=self._data_format)
            y += tf.get_variable('bias', [m], initializer=tf.ones_initializer())
            r, u = tf.split(y, 2, axis=self._feature_axis)
            r, u = tf.sigmoid(r), tf.sigmoid(u)


        with tf.variable_scope('candidate'):
            inputs = tf.concat([x, r * h], axis=self._feature_axis)
            n = channels + self._filters
            m = self._filters
            W = tf.get_variable('kernel', self._kernel + [n, m])
            y = tf.nn.convolution(inputs, W, 'SAME', data_format=self._data_format)
            y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
            h = u * h + (1 - u) * self._activation(y)

        return h, h

def separable_conv1d(inputs,num_outputs,kernel_size,padding='SAME',depth_multiplier=1,*args,**kwargs):
    '''

    :param inputs: [batch_size,W,C]
    :param num_outputs: num
    :param kernel_size: num
    :param padding: 'SAME'/'VALID'
    :return: [batch_size,W1,num_output]
    '''

    if padding == "SAME":
        inputs = tf.pad(inputs,paddings=[[0,0],[kernel_size//2,kernel_size//2],[0,0]])
    inputs = tf.expand_dims(inputs,axis=2)
    res = slim.separable_conv2d(inputs,num_outputs,kernel_size=[kernel_size,1],padding="VALID",
                                depth_multiplier=depth_multiplier,*args,**kwargs)
    return tf.squeeze(res,axis=2)

def probability_adjust(probs,classes=[]):
    if probs.get_shape().ndims == 2:
        return wtfop.probability_adjust(probs=probs,classes=classes)
    else:
        old_shape = tf.shape(probs)
        probs = tf.reshape(probs,[-1,old_shape[-1]])
        out = wtfop.probability_adjust(probs=probs,classes=classes)
        out = tf.reshape(out,old_shape)
        return out

def conv2d_batch_normal(input,decay=0.99,is_training=True,scale=False):
    last_dim_size = input.get_shape().as_list()[-1]
    with tf.variable_scope("BatchNorm"):
        if scale:
            gamma = tf.Variable(tf.ones([last_dim_size],tf.float32), name="gamma")
        else:
            gamma = None
        offset = tf.Variable(tf.zeros([last_dim_size],tf.float32), name="beta")
        moving_collections = ["bn_moving_vars",tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MOVING_AVERAGE_VARIABLES]

        m_mean = tf.Variable(tf.zeros([last_dim_size],tf.float32),trainable=False,name="moving_mean",collections=moving_collections)
        m_variance = tf.Variable(tf.ones([last_dim_size], tf.float32), trainable=False, name="moving_variance",collections=moving_collections)
        if  is_training:
            c_mean, c_variance = tf.nn.moments(input, list(range(len(input.get_shape()) - 1)))
            update_mean =moving_averages.assign_moving_average(m_mean,c_mean,decay)
            update_variance = moving_averages.assign_moving_average(m_variance,c_variance,decay)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,update_mean)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance)
        else:
            c_mean = m_mean
            c_variance = m_variance
        output = tf.nn.batch_normalization(input, c_mean, c_variance, offset, gamma, 1E-6, "BN")
    return output

@add_arg_scope
def group_norm(x, G=32, epsilon=1e-5,weights_regularizer=None,scale=True,offset=True,scope="group_norm"):
    assert scale==True or offset==True
    if x.get_shape().ndims == 4:
        return group_norm_4d(x,G,epsilon,weights_regularizer=weights_regularizer,
                             scale=scale,
                             offset=offset,
                             scope=scope)
    elif x.get_shape().ndims == 2:
        return group_norm_2d(x,G,epsilon,weights_regularizer=weights_regularizer,
                             scale=scale,
                             offset=offset,
                             scope=scope)

@add_arg_scope
def group_norm_4d(x, G=32, epsilon=1e-5,weights_regularizer=None,scale=True,offset=True,scope="group_norm"):
    # x: input features with shape [N,H,W,C]
    # gamma, beta: scale and offset, with shape [1,1,1,C] # G: number of groups for GN
    with tf.variable_scope(scope):
        if x.get_shape().is_fully_defined():
            N,H,W,C = x.get_shape().as_list()
        else:
            none_nr = 0
            for v in x.get_shape().as_list():
                if v is None:
                    none_nr += 1
            if none_nr>1:
                N, H, W, _ = tf.unstack(tf.shape(x),axis=0)
                C = x.get_shape().as_list()[-1]
            else:
                N,H,W,C = x.get_shape().as_list()
        gamma = tf.get_variable(name="gamma",shape=[1,1,1,C],initializer=tf.ones_initializer())
        beta = tf.get_variable(name="beta",shape=[1,1,1,C],initializer=tf.zeros_initializer())
        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(gamma))
        x = wmlt.reshape(x, [N, H, W, G, C // G,])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        if offset:
            x = x-mean
        if scale:
            x = x/ tf.sqrt(var + epsilon)
        x = wmlt.reshape(x, [N,H,W,C])
        return x*gamma + beta

@add_arg_scope
def group_norm_2d(x, G=32, epsilon=1e-5,weights_regularizer=None,scale=True,offset=True,scope="group_norm"):
    with tf.variable_scope(scope):
        N,C = x.get_shape().as_list()
        gamma = tf.get_variable(name="gamma",shape=[1,C],initializer=tf.ones_initializer())
        beta = tf.get_variable(name="beta",shape=[1,C],initializer=tf.zeros_initializer())
        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(gamma))
        x = wmlt.reshape(x, [N,G, C // G,])
        mean, var = tf.nn.moments(x, [2], keep_dims=True)
        if offset:
            x = x-mean
        if scale:
            x = x/tf.sqrt(var + epsilon)
        x = wmlt.reshape(x, [N,C])
        return x*gamma + beta

@add_arg_scope
def group_norm_with_sn(x, G=32, epsilon=1e-5,scope="group_norm_with_sn",sn_iteration=1,max_sigma=None):
    if x.get_shape().ndims == 4:
        return group_norm_4d_with_sn(x,G,epsilon,scope=scope,sn_iteration=sn_iteration,max_sigma=max_sigma)
    elif x.get_shape().ndims == 2:
        return group_norm_2d_with_sn(x,G,epsilon,scope=scope,sn_iteration=sn_iteration,max_sigma=max_sigma)

@add_arg_scope
def group_norm_4d_with_sn(x, G=32, epsilon=1e-5,scope="group_norm_with_sn",sn_iteration=1,max_sigma=None):
    # x: input features with shape [N,H,W,C]
    # gamma, beta: scale and offset, with shape [1,1,1,C] # G: number of groups for GN
    with tf.variable_scope(scope):
        if x.get_shape().is_fully_defined():
            N,H,W,C = x.get_shape().as_list()
        else:
            none_nr = 0
            for v in x.get_shape().as_list():
                if v is None:
                    none_nr += 1
            if none_nr>1:
                N, H, W, _ = tf.unstack(tf.shape(x),axis=0)
                C = x.get_shape().as_list()[-1]
            else:
                N,H,W,C = x.get_shape().as_list()
        gamma = tf.get_variable(name="gamma",shape=[1,1,1,C],initializer=tf.ones_initializer())
        beta = tf.get_variable(name="beta",shape=[1,1,1,C],initializer=tf.zeros_initializer())
        x = wmlt.reshape(x, [N, H, W, G, C // G,])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + epsilon)
        x = wmlt.reshape(x, [N,H,W,C])
        return x*spectral_norm(gamma,sn_iteration,max_sigma=max_sigma) + beta

@add_arg_scope
def group_norm_2d_with_sn(x, G=32, epsilon=1e-5,scope="group_norm_with_sn",sn_iteration=1,max_sigma=None):
    with tf.variable_scope(scope):
        N,C = x.get_shape().as_list()
        gamma = tf.get_variable(name="gamma",shape=[1,C],initializer=tf.ones_initializer())
        beta = tf.get_variable(name="beta",shape=[1,C],initializer=tf.zeros_initializer())
        x = wmlt.reshape(x, [N,G, C // G,])
        mean, var = tf.nn.moments(x, [2], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + epsilon)
        x = wmlt.reshape(x, [N,C])
        return x*spectral_norm(gamma,sn_iteration,max_sigma=max_sigma)+ beta

@add_arg_scope
def layer_norm(x,scope="layer_norm"):
    return tf.contrib.layers.layer_norm(
        inputs=x, begin_norm_axis=-1, begin_params_axis=-1, scope=scope)


@add_arg_scope
def instance_norm(x, eps=1e-5):
    with tf.variable_scope("layer_norm"):
        N, H, W, C = x.shape
        gamma = tf.get_variable(name="gamma",shape=(1,1,1,C),initializer=tf.ones_initializer())
        beta = tf.get_variable(name="beta",shape=(1,1,1,C),initializer=tf.zeros_initializer())
        mean, var = tf.nn.moments(x, [2, 3], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        return x*gamma + beta

def gelu(x):
    cdf = 0.5 * (1.0 + tf.erf(x/ tf.sqrt(2.0)))
    return x* cdf
    #return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

def h_swish(x):
    with tf.name_scope("h_swish"):
        return x*tf.nn.relu6(x+3)/6.0

def mish(x):
    with tf.name_scope("mish"):
        return x*tf.tanh(tf.nn.softplus(x))

@add_arg_scope
def spectral_norm(w, iteration=1,max_sigma=None,debug=False,is_training=True):
    with tf.variable_scope("spectral_norm"):
       w_shape = w.shape.as_list()
       if debug:
           w = tf.reshape(w, [-1, w_shape[-1]])
           s0,_,_ = tf.linalg.svd(w)

       s_w = tf.get_variable("sn_weight",w_shape,initializer=tf.zeros_initializer,trainable=False)
       if is_training:
           s_sigma = tf.get_variable("sigma",(),initializer=tf.ones_initializer(),trainable=False)
           w_shape = w.shape.as_list()
           w = tf.reshape(w, [-1, w_shape[-1]])
           u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
           u_hat = u
           v_hat = None
           for i in range(iteration):
               """
               power iteration
               Usually iteration = 1 will be enough
               """
               v_ = tf.matmul(u_hat, tf.transpose(w))
               v_hat = tf.nn.l2_normalize(v_)

               u_ = tf.matmul(v_hat, w)
               u_hat = tf.nn.l2_normalize(u_)

           u_hat = tf.stop_gradient(u_hat)
           v_hat = tf.stop_gradient(v_hat)

           sigma = tf.reshape(tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat)),())

           with tf.control_dependencies([u.assign(u_hat),s_sigma.assign(sigma)]):
               if max_sigma is None:
                   w_norm = w / sigma
               else:
                   #w_norm = tf.cond(tf.greater(sigma,max_sigma),lambda:w/(sigma/max_sigma),lambda:w,name="scale_w")
                   w_norm = w/(sigma/max_sigma)
               w_norm = tf.reshape(w_norm, w_shape)
           with tf.control_dependencies([s_w.assign(w_norm)]):
               w_norm = tf.identity(w_norm)
       else:
           '''if max_sigma is None:
               w_norm = w / s_sigma
           else:
               #w_norm = tf.cond(tf.greater(sigma,max_sigma),lambda:w/(sigma/max_sigma),lambda:w,name="scale_w")
               w_norm = w/(s_sigma/max_sigma)'''
           w_norm = s_w
       #w_norm = tf.Print(w_norm,[tf.reduce_mean(w_norm),tf.reduce_mean(sigma)],"w_norm")
       if debug:
           w_norm = tf.reshape(w_norm, [-1, w_shape[-1]])
           s,_,_ = tf.linalg.svd(w_norm)
           w_norm = tf.Print(w_norm,[s0[0],s[0]],"Singular1")
           w_norm = tf.reshape(w_norm, w_shape)
       return w_norm

@add_arg_scope
def conv2d_with_sn(inputs,
                   num_outputs,
                   kernel_size,
                   stride=1,
                   padding='SAME',
                   activation_fn=nn.relu,
                   weights_initializer=initializers.xavier_initializer(),
                   weights_regularizer=None,
                   biases_initializer=init_ops.zeros_initializer(),
                   biases_regularizer=None,
                   normalizer_fn=None,
                   normalizer_params=None,
                   outputs_collections=None,
                   rate=1,
                   reuse=None,
                   scope=None,sn_iteration=1):
    del rate
    with variable_scope.variable_scope(scope, 'conv2d', [inputs], reuse=reuse) as sc:
        if isinstance(kernel_size,list):
            shape = kernel_size+[inputs.get_shape().as_list()[-1],num_outputs]
        else:
            shape = [kernel_size,kernel_size,inputs.get_shape().as_list()[-1],num_outputs]
        w = tf.get_variable("kernel", shape=shape,
                            initializer=weights_initializer)
        b = tf.get_variable("bias", [num_outputs], initializer=biases_initializer)
        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(w))
        if biases_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,biases_regularizer(b))

        outputs = tf.nn.conv2d(input=inputs, filter=spectral_norm(w,iteration=sn_iteration), strides=[1, stride, stride, 1],padding=padding) + b
        if normalizer_fn is not None:
            if normalizer_params is None:
                normalizer_params = {}
            outputs = normalizer_fn(outputs,**normalizer_params)
        if activation_fn is not None:
            outputs = utils.collect_named_outputs(outputs_collections, sc.name+"_pre_act", outputs)
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

@add_arg_scope
def depthwise_conv2d_with_sn(inputs,
                   kernel_size,
                   stride=1,
                   padding='SAME',
                   activation_fn=nn.relu,
                   weights_initializer=initializers.xavier_initializer(),
                   weights_regularizer=None,
                   biases_initializer=init_ops.zeros_initializer(),
                   biases_regularizer=None,
                   normalizer_fn=None,
                   normalizer_params=None,
                   outputs_collections=None,
                   rate=1,
                   reuse=None,
                   scope=None,sn_iteration=1):
    del rate
    with variable_scope.variable_scope(scope, 'conv2d', [inputs], reuse=reuse) as sc:
        num_inputs = inputs.get_shape().as_list()[-1]
        if isinstance(kernel_size,list):
            shape = kernel_size+[num_inputs,1]
        else:
            shape = [kernel_size,kernel_size,num_inputs,1]
        w = tf.get_variable("kernel", shape=shape,
                            initializer=weights_initializer)
        b = tf.get_variable("bias", [num_inputs], initializer=biases_initializer)
        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(w))
        if biases_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,biases_regularizer(b))

        outputs = tf.nn.depthwise_conv2d(input=inputs, filter=spectral_norm(w,iteration=sn_iteration), strides=[1, stride, stride, 1],padding=padding) + b
        if normalizer_fn is not None:
            if normalizer_params is None:
                normalizer_params = {}
            outputs = normalizer_fn(outputs,**normalizer_params)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

@add_arg_scope
def fully_connected_with_sn(inputs,
                    num_outputs,
                    activation_fn=nn.relu,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=initializers.xavier_initializer(),
                    weights_regularizer=None,
                    biases_initializer=init_ops.zeros_initializer(),
                    biases_regularizer=None,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    scope=None,sn_iteration=1):
  with tf.variable_scope(scope,
      'fully_connected', [inputs], reuse=reuse) as sc:
      shape = inputs.get_shape().as_list()
      channels = shape[-1]
      w = tf.get_variable("weights", [channels, num_outputs], tf.float32, initializer=weights_initializer,
                          regularizer=weights_regularizer,trainable=trainable,
                          collections=variables_collections)
      if weights_regularizer is not None:
          tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(w))
      if biases_initializer is not None:
          b = tf.get_variable("biases", [num_outputs], tf.float32, initializer=biases_initializer,
                          regularizer=biases_regularizer,trainable=trainable,
                              collections=variables_collections)
          if biases_regularizer is not None:
              tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,biases_regularizer(b))
          outputs = tf.matmul(inputs,spectral_norm(w,iteration=sn_iteration))+b
      else:
          outputs = tf.matmul(inputs,spectral_norm(w,iteration=sn_iteration))
      # Apply normalizer function / layer.
      if normalizer_fn is not None:
          if not normalizer_params:
              normalizer_params = {}
          outputs = normalizer_fn(outputs, **normalizer_params)

      if activation_fn is not None:
          outputs = activation_fn(outputs)

      return utils.collect_named_outputs(outputs_collections, sc.name, outputs)




def orthogonal_initializer(shape, dtype=tf.float32, *args, **kwargs):
  """Generates orthonormal matrices with random values.

  Orthonormal initialization is important for RNNs:
    http://arxiv.org/abs/1312.6120
    http://smerity.com/articles/2016/orthogonal_init.html

  For non-square shapes the returned matrix will be semi-orthonormal: if the
  number of columns exceeds the number of rows, then the rows are orthonormal
  vectors; but if the number of rows exceeds the number of columns, then the
  columns are orthonormal vectors.

  We use SVD decomposition to generate an orthonormal matrix with random
  values. The same way as it is done in the Lasagne library for Theano. Note
  that both u and v returned by the svd are orthogonal and random. We just need
  to pick one with the right shape.

  Args:
    shape: a shape of the tensor matrix to initialize.
    dtype: a dtype of the initialized tensor.
    *args: not used.
    **kwargs: not used.

  Returns:
    An initialized tensor.
  """
  del args
  del kwargs
  flat_shape = (shape[0], np.prod(shape[1:]))
  w = np.random.randn(*flat_shape)
  u, _, v = np.linalg.svd(w, full_matrices=False)
  w = u if u.shape == flat_shape else v
  return tf.constant(w.reshape(shape), dtype=dtype)

def non_local_block(net,multiplier=0.5,n_head=1,keep_prob=None,is_training=False,scope=None):
    def reshape_net(net):
        shape = net.get_shape().as_list()
        new_shape = [-1,shape[1]*shape[2],shape[3]]
        net = tf.reshape(net,new_shape)
        return net
    def restore_shape(net,shape,channel):
        out_shape = [-1,shape[1],shape[2],channel]
        net = tf.reshape(net,out_shape)
        return net

    with tf.variable_scope(scope,default_name="non_local"):
        shape = net.get_shape().as_list()
        channel = shape[-1]
        m_channel = int(channel*multiplier)
        Q = slim.conv2d(net,m_channel,[1,1],activation_fn=None)
        K = slim.conv2d(net,m_channel,[1,1],activation_fn=None)
        V = slim.conv2d(net,m_channel,[1,1],activation_fn=None)
        Q = reshape_net(Q)
        K = reshape_net(K)
        V = reshape_net(V)
        out = nlpl.multi_head_attention(Q, K, V, n_head=n_head,keep_prob=keep_prob, is_training=is_training,
                                 use_mask=False)
        out = restore_shape(out,shape,m_channel)
        out = slim.conv2d(out,channel,[1,1],activation_fn=None,
                          weights_initializer=tf.zeros_initializer)
        return net+out

def non_local_blockv1(net,inner_dims_multiplier=[8,8,2],n_head=1,keep_prob=None,is_training=False,scope=None,
                      conv_op=slim.conv2d,pool_op=None,normalizer_fn=slim.batch_norm,normalizer_params=None,
                      gamma_initializer=tf.constant_initializer(0.0)):
    def reshape_net(net):
        shape = wmlt.combined_static_and_dynamic_shape(net)
        new_shape = [shape[0],shape[1]*shape[2],shape[3]]
        net = tf.reshape(net,new_shape)
        return net
    def restore_shape(net,shape,channel):
        out_shape = [shape[0],shape[1],shape[2],channel]
        net = tf.reshape(net,out_shape)
        return net

    if isinstance(inner_dims_multiplier,int):
        inner_dims_multiplier = [inner_dims_multiplier]
    if len(inner_dims_multiplier) == 1:
        inner_dims_multiplier = inner_dims_multiplier*3

    with tf.variable_scope(scope,default_name="non_local"):
        shape = wmlt.combined_static_and_dynamic_shape(net)
        channel = shape[-1]
        m_channelq = channel//inner_dims_multiplier[0]
        m_channelk = channel//inner_dims_multiplier[1]
        m_channelv = channel//inner_dims_multiplier[2]

        Q = conv_op(net,m_channelq,[1,1],activation_fn=None,normalizer_fn=None,scope="q_conv")
        K = conv_op(net,m_channelk,[1,1],activation_fn=None,normalizer_fn=None,scope="k_conv")
        V = conv_op(net,m_channelv,[1,1],activation_fn=None,normalizer_fn=None,scope="v_conv")
        if pool_op is not None:
            K = pool_op(K,kernel_size=2, stride=2,padding="SAME")
            V = pool_op(V,kernel_size=2, stride=2,padding="SAME")
        Q = reshape_net(Q)
        K = reshape_net(K)
        V = reshape_net(V)
        out = nlpl.multi_head_attention(Q, K, V, n_head=n_head,keep_prob=keep_prob, is_training=is_training,
                                        use_mask=False)
        out = restore_shape(out,shape,m_channelv)
        out = conv_op(out,channel,[1,1],normalizer_fn=None,scope="attn_conv")
        normalizer_params  = normalizer_params or {}
        if normalizer_fn is not None:
            out = normalizer_fn(out+net,**normalizer_params)
        gamma = tf.get_variable("gamma", [1], initializer=gamma_initializer)
        return gamma*out+net

def non_local_blockv2(net,multiplier=0.5,n_head=1,keep_prob=None,is_training=False,scope=None,normalizer_fn=slim.batch_norm,normalizer_params=None):
    def reshape_net(net):
        shape = net.get_shape().as_list()
        new_shape = [-1,shape[1]*shape[2],shape[3]]
        net = tf.reshape(net,new_shape)
        return net
    def restore_shape(net,shape,channel):
        out_shape = [-1,shape[1],shape[2],channel]
        net = tf.reshape(net,out_shape)
        return net

    def pos_embedding_fn(V,P):
        v_shape = tf.shape(V)
        V = nlpl.split_states(V,n_head)
        V = V+P
        V = tf.reshape(V,v_shape)
        return V

    with tf.variable_scope(scope,default_name="non_local"):
        shape = net.get_shape().as_list()
        channel = shape[-1]
        m_channel = int(channel*multiplier)
        Q = slim.conv2d(net,m_channel,[1,1],activation_fn=None,normalizer_fn=None)
        K = slim.conv2d(net,m_channel,[1,1],activation_fn=None,normalizer_fn=None)
        V = slim.conv2d(net,m_channel,[1,1],activation_fn=None,normalizer_fn=None)
        pos_embedding = tf.get_variable("pos_embedding",shape=[1]+Q.get_shape().as_list()[1:3]+[n_head,1],initializer=tf.random_normal_initializer()),
        Q = pos_embedding_fn(Q,pos_embedding)
        K = pos_embedding_fn(K,pos_embedding)
        Q = reshape_net(Q)
        K = reshape_net(K)
        V = reshape_net(V)
        out = nlpl.multi_head_attention(Q, K, V, n_head=n_head,keep_prob=keep_prob, is_training=is_training,
                                        use_mask=False)
        out = restore_shape(out,shape,m_channel)
        out = slim.conv2d(out,channel,[1,1],normalizer_fn=None)
        normalizer_params  = normalizer_params or {}
        out = normalizer_fn(out+net,**normalizer_params)
        return out

def augmented_conv2d(net,Fout,dv,kernel=[3,3],n_head=1,keep_prob=None,is_training=False,scope=None,normalizer_fn=slim.batch_norm,normalizer_params=None):
    def reshape_net(net):
        shape = net.get_shape().as_list()
        new_shape = [-1,shape[1]*shape[2],shape[3]]
        net = tf.reshape(net,new_shape)
        return net
    def restore_shape(net,shape,channel):
        out_shape = [-1,shape[1],shape[2],channel]
        net = tf.reshape(net,out_shape)
        return net

    with tf.variable_scope(scope,default_name="augmented_conv2d"):
        conv_out = slim.conv2d(net,Fout-dv,kernel)

        shape = net.get_shape().as_list()
        channel = shape[-1]
        m_channel = dv
        Q = slim.conv2d(net,m_channel,[1,1],activation_fn=None,normalizer_fn=None)
        K = slim.conv2d(net,m_channel,[1,1],activation_fn=None,normalizer_fn=None)
        V = slim.conv2d(net,m_channel,[1,1],activation_fn=None,normalizer_fn=None)
        Q = reshape_net(Q)
        K = reshape_net(K)
        V = reshape_net(V)
        out = nlpl.relative_multi_head_attention(Q, K, V, n_head=n_head,keep_prob=keep_prob, is_training=is_training,
                                        use_mask=False)
        out = restore_shape(out,shape,m_channel)
        out = slim.conv2d(out,channel,[1,1],normalizer_fn=None)
        normalizer_params  = normalizer_params or {}
        out = normalizer_fn(out+net,**normalizer_params)
        out = tf.concat([conv_out,out],axis=3)
        return out

def cnn_self_attenation(net,channel=None,n_head=1,keep_prob=None,is_training=False):
    old_channel = net.get_shape().as_list()[-1]
    if channel is not None:
        net = slim.conv2d(net,channel,[1,1],scope="projection_0")
    shape = net.get_shape().as_list()
    new_shape = [-1,shape[1]*shape[2],shape[3]]
    net = tf.reshape(net,new_shape)
    net = nlpl.self_attenation(net,n_head=n_head,keep_prob=keep_prob,is_training=is_training,use_mask=False)
    if channel is not None:
        out_shape = [-1,shape[1],shape[2],channel]
        net = tf.reshape(net,out_shape)
        net = slim.conv2d(net,old_channel,[1,1],scope="projection_1")
    else:
        out_shape = [-1,shape[1],shape[2],old_channel]
        net = tf.reshape(net,out_shape)
    return net

def cnn_self_hattenation(net,channel=None,n_head=1,keep_prob=None,is_training=False):
    old_channel = net.get_shape().as_list()[-1]
    if channel is not None:
        net = slim.conv2d(net,channel,[1,1],scope="projection_0")
    shape = wmlt.combined_static_and_dynamic_shape(net)
    new_shape = [shape[0]*shape[1],shape[2],shape[3]]
    net = tf.reshape(net,new_shape)
    net = nlpl.self_attenation(net,n_head=n_head,keep_prob=keep_prob,is_training=is_training,use_mask=False)
    if channel is not None:
        out_shape = [-1,shape[1],shape[2],channel]
        net = tf.reshape(net,out_shape)
        net = slim.conv2d(net,old_channel,[1,1],scope="projection_1")
    else:
        out_shape = [-1,shape[1],shape[2],old_channel]
        net = tf.reshape(net,out_shape)
    return net

def cnn_self_vattenation(net,channel=None,n_head=1,keep_prob=None,is_training=False):
    net = tf.transpose(net,perm=[0,2,1,3],name="transpose_0")
    output = cnn_self_hattenation(net,channel,n_head,keep_prob,is_training=is_training)
    return tf.transpose(output,perm=[0,2,1,3],name="transpose_1")

'''
GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond
'''
def gc_block(net,r=16,normalizer_fn=slim.layers.layer_norm,normalizer_params=None,scope=None,conv_op=slim.conv2d):
    with tf.variable_scope(scope,default_name="gc_block"):
        (batch_size,height,width,channel) = wmlt.combined_static_and_dynamic_shape(net)
        input_x = tf.reshape(net,[batch_size,height*width,channel])
        input_x = tf.transpose(input_x,perm=(0,2,1))
        #input_x:[B,C,WH]
        context_x = conv_op(net,1,[1,1],activation_fn=None,normalizer_fn=None)
        context_x = tf.reshape(context_x,[batch_size,height*width])
        #context_x:[B,WH]
        context_mask = tf.nn.softmax(context_x)
        context_mask = tf.expand_dims(context_mask,axis=-1)
        #context_x:[B,WH,1]
        context = tf.matmul(input_x,context_mask)
        #context:[B,C,1]
        context = tf.transpose(context,perm=(0,2,1))
        #context:[B,1,C]
        context = tf.expand_dims(context,axis=1)
        #context:[B,1,1,C]
        context = conv_op(context,channel//r,[1,1],activation_fn=tf.nn.relu,normalizer_fn=normalizer_fn,normalizer_params=normalizer_params)
        #context:[B,1,1,C//r]
        context = conv_op(context,channel,[1,1],activation_fn=None,normalizer_fn=None)
        #context:[B,1,1,C]
        return context+net

'''
Squeeze-and-Excitation Networks
'''
def se_block(net,r=16,fc_op=slim.fully_connected,activation_fn=tf.nn.relu,scope=None,summary_fn=None):
    with tf.variable_scope(scope,"SEBlock"):
        channel = net.get_shape().as_list()[-1]
        mid_channel = channel//r
        org_net = net
        net = tf.reduce_mean(net,axis=[1,2],keepdims=False)
        net = fc_op(net,mid_channel,activation_fn=activation_fn,normalizer_fn=None)
        net = fc_op(net,channel,activation_fn=tf.nn.sigmoid,normalizer_fn=None)
        if summary_fn is not None:
            summary_fn("se_sigmoid_value",net)
        net = tf.expand_dims(net,axis=1)
        net = tf.expand_dims(net,axis=1)
        return net*org_net

def dropblock(inputs,keep_prob,is_training,block_size=7,scope=None,seed=int(time.time()),all_channel=False):
    with tf.variable_scope(scope,default_name="dropblock"):
        if not is_training:
            return tf.identity(inputs)
        drop_prob = (1.0-keep_prob)/(block_size*block_size)
        if all_channel:
            mask = tf.random_uniform(shape=tf.shape(inputs)[:-1],minval=0.,maxval=1.0,dtype=tf.float32,seed=seed)
            mask = tf.expand_dims(mask,axis=-1)
        else:
            mask = tf.random_uniform(shape=tf.shape(inputs),minval=0.,maxval=1.0,dtype=tf.float32,seed=seed)
        bin_mask = tf.greater(mask,drop_prob)
        bin_mask = tf.cast(bin_mask,tf.float32)
        nozero = tf.reduce_sum(bin_mask)
        allsize = tf.cast(tf.reduce_prod(tf.shape(bin_mask)),tf.float32)
        ratio = allsize/nozero
        bin_mask = -tf.nn.max_pool(-bin_mask,ksize=[1,block_size,block_size,1],strides=[1,1,1,1],padding="SAME")
        bin_mask = bin_mask*ratio
        bin_mask = tf.stop_gradient(bin_mask)
        return inputs*bin_mask


def min_pool2d(inputs,*args,**kwargs):
    return -slim.max_pool2d(-inputs,*args,**kwargs)

def orthogonal_regularizer(scale=1e-4) :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Conv layer as kernel regularizer"""

    def ortho_reg(w) :
        with tf.name_scope("orthogonal_regularizer"):
            w = tf.convert_to_tensor(w)
            """ Reshaping the matrxi in to 2D tensor for enforcing orthogonality"""
            c = w.get_shape().as_list()[-1]

            w = tf.reshape(w, [-1, c])

            """ Declaring a Identity Tensor of appropriate size"""
            identity = tf.eye(c)

            """ Regularizer Wt*W - I """
            w_mul = tf.matmul(w, w,transpose_a=True)
            reg = tf.subtract(w_mul, identity)
            """Calculating the Loss Obtained"""
            ortho_loss = tf.nn.l2_loss(reg)

            return scale * ortho_loss

    return ortho_reg

def orthogonal_regularizerv2(scale=1e-4) :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Conv layer as kernel regularizer"""

    def ortho_reg(w) :
        with tf.name_scope("orthogonal_regularizerv2"):
            """ Reshaping the matrxi in to 2D tensor for enforcing orthogonality"""
            w = tf.convert_to_tensor(w)
            c = w.get_shape().as_list()[-1]

            w = tf.reshape(w, [-1, c])

            """ Declaring a Identity Tensor of appropriate size"""
            identity = tf.eye(c)

            """ Regularizer Wt*W(1 - I) """
            w_mul = tf.matmul(w, w,transpose_a=True)
            mask = tf.ones_like(identity)-identity
            reg = mask*w_mul

            """Calculating the Loss Obtained"""
            ortho_loss = tf.nn.l2_loss(reg)

            return scale * ortho_loss

    return ortho_reg


scale_gradient_value = 0.5
def scale_gradient(x,scale):
    global scale_gradient_value
    scale_gradient_value = scale
    with tf.name_scope("scale_gradient"):
        return _scale_gradient(x)

@tf.custom_gradient
def _scale_gradient(x):
    def grad(x):
        return tf.ones_like(x)*scale_gradient_value
    return tf.identity(x),grad
