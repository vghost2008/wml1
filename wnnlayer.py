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
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import numpy as np
import time
from tensorflow.contrib.framework.python.ops import add_arg_scope
import basic_tftools as btf
from collections import Iterable
import wsummary

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
def group_norm(x, G=32, epsilon=1e-5,weights_regularizer=None,scale=True,offset=True,sub_mean=True,scope="group_norm",dtype=None):
    assert scale==True or offset==True
    C = x.get_shape().as_list()[-1]
    assert C%G==0,f"Unmatch channel {C} and group {G} size"
    if x.get_shape().ndims == 4:
        return group_norm_4d_v1(x,G,epsilon,weights_regularizer=weights_regularizer,
                             scale=scale,
                             offset=offset,
                             sub_mean=sub_mean,
                             scope=scope,dtype=dtype)
    elif x.get_shape().ndims == 2:
        return group_norm_2d(x,G,epsilon,weights_regularizer=weights_regularizer,
                             scale=scale,
                             offset=offset,
                             sub_mean=sub_mean,
                             scope=scope,dtype=dtype)
    else:
        raise NotImplementedError

@add_arg_scope
def group_norm_4d_v1(x, G=32, epsilon=1e-5,weights_regularizer=None,scale=True,offset=True,sub_mean=True,scope="group_norm",dtype=None):
    # x: input features with shape [N,H,W,C]
    # gamma, beta: scale and offset, with shape [1,1,1,C] # G: number of groups for GN
    with tf.variable_scope(scope):
        N,H,W,C = btf.combined_static_and_dynamic_shape(x)
        gamma = tf.get_variable(name="gamma",shape=[1,1,1,G,C//G],initializer=tf.ones_initializer(),dtype=dtype)
        if offset:
            beta = tf.get_variable(name="beta",shape=[1,1,1,G,C//G],initializer=tf.zeros_initializer(),dtype=dtype)
        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(gamma))
        x = wmlt.reshape(x, [N, H, W, G, C // G,])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)

        gain = tf.math.rsqrt(var + epsilon)
        if sub_mean:
            offset_value = -mean * gain
        else:
            offset_value = None

        if scale:
            gain *= gamma
            if offset_value is not None:
                offset_value *= gamma

        if offset:
            if offset_value is None:
                offset_value = beta
            else:
                offset_value += beta

        if offset_value is not None:
            x = x * gain + offset_value
        else:
            x = offset_value
        x = wmlt.reshape(x, [N,H,W,C])
        return x

@add_arg_scope
def group_norm_4d_v0(x, G=32, epsilon=1e-5,weights_regularizer=None,scale=True,offset=True,scope="group_norm"):
    # x: input features with shape [N,H,W,C]
    # gamma, beta: scale and offset, with shape [1,1,1,C] # G: number of groups for GN
    with tf.variable_scope(scope):
        N,H,W,C = btf.combined_static_and_dynamic_shape(x)
        gamma = tf.get_variable(name="gamma",shape=[1,1,1,C],initializer=tf.ones_initializer())
        gamma = tf.reshape(gamma,[1,1,1,G,C//G])

        beta = tf.get_variable(name="beta",shape=[1,1,1,C],initializer=tf.zeros_initializer())
        beta = tf.reshape(beta,[1,1,1,G,C//G])

        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(gamma))
        x = wmlt.reshape(x, [N, H, W, G, C // G,])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)

        gain = tf.math.rsqrt(var + epsilon)
        if offset:
            offset_value = -mean * gain
        else:
            offset_value = tf.zeros_like(beta)

        if scale:
            gain *= gamma
            offset_value *= gamma

        if offset:
            offset_value += beta

        x = x * gain + offset_value
        x = wmlt.reshape(x, [N,H,W,C])
        return x

@add_arg_scope
def group_norm_2d(x, G=32, epsilon=1e-5,weights_regularizer=None,scale=True,offset=True,sub_mean=True,scope="group_norm",dtype=None):
    with tf.variable_scope(scope):
        N,C = x.get_shape().as_list()
        gamma = tf.get_variable(name="gamma",shape=[1,G,C//G],initializer=tf.ones_initializer(),dtype=dtype)
        if offset:
            beta = tf.get_variable(name="beta",shape=[1,G,C//G],initializer=tf.zeros_initializer(),dtype=dtype)
        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(gamma))
        x = wmlt.reshape(x, [N,G, C // G,])
        mean, var = tf.nn.moments(x, [2], keep_dims=True)
        gain = tf.math.rsqrt(var+epsilon)
        if sub_mean:
            offset_value = -mean*gain
        else:
            offset_value = None
        if scale:
            gain *= gamma
            if offset_value is not None:
                offset_value *= gamma
        if offset:
            if offset_value is not None:
                offset_value += beta
            else:
                offset_value = beta
        x = x*gain+offset_value
        x = wmlt.reshape(x, [N,C])
        return x
    
@add_arg_scope
def group_norm_2d_v0(x, G=32, epsilon=1e-5,weights_regularizer=None,scale=True,offset=True,scope="group_norm"):
    with tf.variable_scope(scope):
        N,C = x.get_shape().as_list()
        gamma = tf.get_variable(name="gamma",shape=[1,C],initializer=tf.ones_initializer())
        beta = tf.get_variable(name="beta",shape=[1,C],initializer=tf.zeros_initializer())
        gamma = tf.reshape(gamma,[1,G,C//G])
        beta = tf.reshape(beta,[1,G,C//G])
        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(gamma))
        x = wmlt.reshape(x, [N,G, C // G,])
        mean, var = tf.nn.moments(x, [2], keep_dims=True)
        gain = tf.math.rsqrt(var+epsilon)
        offset_value = -mean*gain
        if scale:
            gain *= gamma
            offset_value *= gamma
        if offset:
            offset += beta
        x = x*gain+offset_value
        x = wmlt.reshape(x, [N,C])
        return x

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
def group_norm_v2(x, gamma=None,beta=None,G=32, epsilon=1e-5,weights_regularizer=None,scale=True,offset=True,sub_mean=True,scope=None,dtype=None):
    # x: input features with shape [N,H,W,C]
    # gamma, beta: scale and offset, with shape [1,1,1,C] # G: number of groups for GN
    with tf.variable_scope(scope,default_name="group_norm_v2"):
        N,H,W,C = btf.combined_static_and_dynamic_shape(x)
        if gamma is not None:
            gamma = tf.reshape(gamma,shape=[N,1,1,G,C//G])
        if beta is not None:
            beta = tf.reshape(beta,shape=[N,1,1,G,C//G])
        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(gamma))
        x = wmlt.reshape(x, [N, H, W, G, C // G,])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)

        gain = tf.math.rsqrt(var + epsilon)
        if sub_mean:
            offset_value = -mean * gain
        else:
            offset_value = None

        if scale:
            gain *= gamma
            if offset_value is not None:
                offset_value *= gamma

        if offset:
            if offset_value is None:
                offset_value = beta
            else:
                offset_value += beta

        if offset_value is not None:
            x = x * gain + offset_value
        else:
            x = offset_value
        x = wmlt.reshape(x, [N,H,W,C])
        return x

@add_arg_scope
def evo_norm_s0(x,*args,**kwargs):
    if len(x.get_shape()) == 4:
        return evo_norm_s0_4d(x,*args,**kwargs)
    elif len(x.get_shape())==2:
        return evo_norm_s0_2d(x,*args,**kwargs)
    else:
        raise NotImplementedError(f"Input dims must be 2 or 4.")

@add_arg_scope
def evo_norm_s0_4d(x, G=32, epsilon=1e-5,weights_regularizer=None,scale=True,scope="evo_norm_s0"):
    # x: input features with shape [N,H,W,C]
    # gamma, beta: scale with shape [1,1,1,C] # G: number of groups for GN
    # WARNING: after evo_norm_s0, there is no need to append a activation fn.
    with tf.variable_scope(scope):
        N,H,W,C = btf.combined_static_and_dynamic_shape(x)
        gamma = tf.get_variable(name="gamma",shape=[1,1,1,G,C//G],initializer=tf.ones_initializer())
        beta = tf.get_variable(name="beta",shape=[1,1,1,G,C//G],initializer=tf.zeros_initializer())
        v1 = tf.get_variable(name="v1",shape=[1,1,1,G,C//G],initializer=tf.ones_initializer())
        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(gamma))
        assert C%G==0,f"Error C={C}, G={G}"
        x = wmlt.reshape(x, [N, H, W, G, C // G,])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)

        gain = tf.math.rsqrt(var + epsilon)

        if scale:
            gain *= gamma

        x = x * tf.nn.sigmoid(x*v1)*gain + beta
        x = wmlt.reshape(x, [N,H,W,C])
        return x

@add_arg_scope
def evo_norm_s0_2d(x, G=32, epsilon=1e-5,weights_regularizer=None,scale=True,scope="evo_norm_s0"):
    # x: input features with shape [N,H,W,C]
    # gamma, beta: scale with shape [1,1,1,C] # G: number of groups for GN
    # WARNING: after evo_norm_s0, there is no need to append a activation fn.
    with tf.variable_scope(scope):
        N,C = btf.combined_static_and_dynamic_shape(x)
        gamma = tf.get_variable(name="gamma",shape=[1,G,C//G],initializer=tf.ones_initializer())
        beta = tf.get_variable(name="beta",shape=[1,G,C//G],initializer=tf.zeros_initializer())
        v1 = tf.get_variable(name="v1",shape=[1,G,C//G],initializer=tf.ones_initializer())
        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(gamma))
        assert C%G==0,f"Error C={C}, G={G}"
        x = wmlt.reshape(x, [N, G, C // G,])
        mean, var = tf.nn.moments(x, [2], keep_dims=True)

        gain = tf.math.rsqrt(var + epsilon)

        if scale:
            gain *= gamma

        x = x * tf.nn.sigmoid(x*v1)*gain + beta
        x = wmlt.reshape(x, [N,C])
        return x

@add_arg_scope
def spectral_norm_for_conv(x,is_training=True,scope=None):
    #用于normalizer的conv2d, full_connected的权重名为weights并且需要与本函数在同一个value_scope中
    def get_weights():
        ws = tf.trainable_variables(tf.get_variable_scope().name)
        res = None
        total_nr = 0
        for w in ws:
            if w.name.endswith("weights:0"):
                res = w
                total_nr += 1
        assert total_nr==1,"error weights nr."
        return res

    w = get_weights()
    B,H,W,C = x.get_shape().as_list()

    with tf.variable_scope(scope,"spectral_norm"):
        beta = tf.get_variable(name="beta", shape=[1, 1, 1, C], initializer=tf.ones_initializer())
        gamma = tf.get_variable(name="gamma", shape=[1, 1, 1, C], initializer=tf.ones_initializer())
        s_sigma = tf.get_variable("sigma", (), initializer=tf.ones_initializer(), trainable=not is_training)

        if is_training:
            w_shape = w.shape.as_list()
            w = tf.reshape(w, [-1, w_shape[-1]])
            u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
            with tf.name_scope("get_sigma"):
                u_hat = u
                #one iterator
                v_ = tf.matmul(u_hat, tf.transpose(w))
                v_hat = tf.nn.l2_normalize(v_)

                u_ = tf.matmul(v_hat, w)
                u_hat = tf.nn.l2_normalize(u_)

                u_hat = tf.stop_gradient(u_hat)
                v_hat = tf.stop_gradient(v_hat)

                sigma = tf.reshape(tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat)),())

            with tf.control_dependencies([u.assign(u_hat),s_sigma.assign(sigma)]):
                v = gamma/sigma
                x = x*v+beta
        else:
            v = gamma /s_sigma
            x = x * v+beta

        return x

@add_arg_scope
def layer_norm(x,scope="layer_norm"):
    return tf.contrib.layers.layer_norm(
        inputs=x, begin_norm_axis=-1, begin_params_axis=-1, scope=scope)

def graph_norm(input,decay=0.99,scale=True):
    last_dim_size = input.get_shape().as_list()[-1]
    with tf.variable_scope("BatchNorm"):
        if scale:
            gamma = tf.get_variable(name="gamma",shape=(last_dim_size),initializer=tf.ones_initializer())
        else:
            gamma = None
        offset = tf.get_variable(name="beta",shape=(last_dim_size),initializer=tf.zeros_initializer())
        c_mean, c_variance = tf.nn.moments(input, list(range(len(input.get_shape()) - 1)))
        '''wsummary.variable_summaries_v2(c_mean, "graph_norm_mean", "layer_norm")
        wsummary.variable_summaries_v2(c_variance, "graph_norm_variance", "layer_norm")
        wsummary.variable_summaries_v2(input, "net", "net")'''
        output = tf.nn.batch_normalization(input, c_mean, c_variance, offset, gamma, 1E-6, "BN")
    return output

def group_graph_norm(x,G=16,decay=0.99,scale=True,offset=True,epsilon=1e-8):
    with tf.variable_scope("group_graph_norm"):
        dtype = x.dtype
        N,C = wmlt.combined_static_and_dynamic_shape(x)
        gamma = tf.get_variable(name="gamma",shape=[1,G,C//G],initializer=tf.ones_initializer(),dtype=dtype)
        if offset:
            beta = tf.get_variable(name="beta",shape=[1,G,C//G],initializer=tf.zeros_initializer(),dtype=dtype)
        x = wmlt.reshape(x, [N,G, C // G,])
        mean, var = tf.nn.moments(x, [0,2], keep_dims=True)
        gain = tf.math.rsqrt(var+epsilon)
        offset_value = -mean*gain
        if scale:
            gain *= gamma
            if offset_value is not None:
                offset_value *= gamma
        if offset:
            if offset_value is not None:
                offset_value += beta
            else:
                offset_value = beta
        x = x*gain+offset_value
        x = wmlt.reshape(x, [N,C])
        return x
    return output

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
def spectral_norm(w, iteration=1,max_sigma=None,is_training=True,scope=None,dtype=None):
    with tf.variable_scope(scope,"spectral_norm"):
       w_shape = w.shape.as_list()

       s_w = tf.get_variable("sn_weight",w_shape,initializer=tf.zeros_initializer,trainable=False,dtype=dtype)
       if is_training:
           s_sigma = tf.get_variable("sigma",(),initializer=tf.ones_initializer(),trainable=False,dtype=dtype)
           w_shape = w.shape.as_list()
           w = tf.reshape(w, [-1, w_shape[-1]])
           u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False,dtype=dtype)
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
                   w_norm = w/(sigma/max_sigma)
               w_norm = tf.reshape(w_norm, w_shape)
           with tf.control_dependencies([s_w.assign(w_norm)]):
               w_norm = tf.identity(w_norm)
       else:
           w_norm = s_w

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
                   scope=None,sn_iteration=1,dtype=None):
    del rate
    print(f"conv2d_with_sn is deprecated.")
    with variable_scope.variable_scope(scope, 'conv2d', [inputs], reuse=reuse) as sc:
        if isinstance(kernel_size,list):
            shape = kernel_size+[inputs.get_shape().as_list()[-1],num_outputs]
        else:
            shape = [kernel_size,kernel_size,inputs.get_shape().as_list()[-1],num_outputs]
        w = tf.get_variable("kernel", shape=shape,
                            initializer=weights_initializer,dtype=dtype)
        b = tf.get_variable("bias", [num_outputs], initializer=biases_initializer,dtype=dtype)
        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(w))
        if biases_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,biases_regularizer(b))

        outputs = tf.nn.conv2d(input=inputs, filter=spectral_norm(w,iteration=sn_iteration,dtype=dtype),
                               strides=[1, stride, stride, 1],
                               padding=padding) + b
        if normalizer_fn is not None:
            if normalizer_params is None:
                normalizer_params = {}
            outputs = normalizer_fn(outputs,**normalizer_params)
        if activation_fn is not None:
            outputs = utils.collect_named_outputs(outputs_collections, sc.name+"_pre_act", outputs)
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

@add_arg_scope
def conv2d_with_sn_v2(inputs,
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
        if biases_initializer is not None and normalizer_fn is None:
            b = tf.get_variable("bias", [num_outputs], initializer=biases_initializer)
        else:
            b = None
        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(w))
        if b is not None and biases_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,biases_regularizer(b))

        outputs = tf.nn.conv2d(input=inputs, filter=spectral_norm(w,iteration=sn_iteration), strides=[1, stride, stride, 1],padding=padding)
        if b is not None:
            outputs = outputs + b
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
        if biases_initializer is not None and normalizer_fn is None:
            b = tf.get_variable("bias", [num_inputs], initializer=biases_initializer)
        else:
            b = None
        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(w))
        if b is not None and biases_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,biases_regularizer(b))

        outputs = tf.nn.depthwise_conv2d(input=inputs, filter=spectral_norm(w,iteration=sn_iteration),
                                         strides=[1, stride, stride, 1],padding=padding)
        if normalizer_fn is not None:
            if normalizer_params is None:
                normalizer_params = {}
            outputs = normalizer_fn(outputs,**normalizer_params)
        elif b is not None:
            outputs = outputs + tf.reshape(b,[1,1,1,num_inputs])

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        if outputs_collections is not None:
            return utils.collect_named_outputs(outputs_collections, sc.name, outputs)
        else:
            return outputs

@add_arg_scope
def separable_conv2d_with_sn(inputs,
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
                   scope=None):
    with tf.variable_scope(scope,default_name="separable_conv2d_with_sn",reuse=reuse) as sc:
        net = depthwise_conv2d_with_sn(inputs,kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       activation_fn=None,
                                       weights_initializer=weights_initializer,
                                       weights_regularizer=weights_regularizer,
                                       biases_initializer=None,
                                       normalizer_fn=None,
                                       outputs_collections=outputs_collections,
                                       rate=rate,
                                       )
        if num_outputs is not None:
            net = conv2d_with_sn(net,num_outputs,kernel_size=[1,1],
                                 padding="SAME",
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 weights_initializer=weights_initializer,
                                 outputs_collections=outputs_collections)

        b = None
        if normalizer_fn is not None:
            if normalizer_params is None:
                normalizer_params = {}
            net = normalizer_fn(net, **normalizer_params)
        else:
            if biases_initializer is not None:
                b = tf.get_variable("bias", [num_outputs], initializer=biases_initializer)
            if b is not None and biases_regularizer is not None:
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,biases_regularizer(b))
            if b is not None:
                b = tf.reshape(b,[1,1,1,num_outputs])
                net = net + b

        if activation_fn is not None:
            net = activation_fn(net)

        if outputs_collections is not None:
            return utils.collect_named_outputs(outputs_collections, sc.name, net)
        else:
            return net


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
                    scope=None,sn_iteration=1,dtype=None):
  with tf.variable_scope(scope,
      'fully_connected', [inputs], reuse=reuse) as sc:
      shape = inputs.get_shape().as_list()
      channels = shape[-1]
      w = tf.get_variable("weights", [channels, num_outputs],
                          initializer=weights_initializer,
                          regularizer=weights_regularizer,trainable=trainable,
                          collections=variables_collections,dtype=dtype)
      if weights_regularizer is not None:
          tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(w))
      if biases_initializer is not None:
          b = tf.get_variable("biases", [num_outputs],
                              initializer=biases_initializer,
                              regularizer=biases_regularizer,trainable=trainable,
                              collections=variables_collections,dtype=dtype)
          if biases_regularizer is not None:
              tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,biases_regularizer(b))
          outputs = tf.matmul(inputs,spectral_norm(w,iteration=sn_iteration,dtype=dtype))+b
      else:
          outputs = tf.matmul(inputs,spectral_norm(w,iteration=sn_iteration,dtype=dtype))
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

def non_local_blockv1(net,inner_dims_multiplier=[8,8,2],
                      inner_dims=None,
                      n_head=1,keep_prob=None,is_training=False,scope=None,
                      conv_op=slim.conv2d,pool_op=None,normalizer_fn=slim.batch_norm,normalizer_params=None,
                      activation_fn=tf.nn.relu,
                      gamma_initializer=tf.constant_initializer(0.0),reuse=None,
                      weighed_sum=True):
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
    if inner_dims is not None:
        if isinstance(inner_dims, int):
            inner_dims = [inner_dims]
        if len(inner_dims) == 1:
            inner_dims = inner_dims*3

    with tf.variable_scope(scope,default_name="non_local",reuse=reuse):
        shape = wmlt.combined_static_and_dynamic_shape(net)
        channel = shape[-1]
        if inner_dims is not None:
            m_channelq = inner_dims[0]
            m_channelk = inner_dims[1]
            m_channelv = inner_dims[2]
            pass
        else:
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
        out = conv_op(out,channel,[1,1],
                      activation_fn=None,
                      normalizer_fn=None,
                      scope="attn_conv")
        normalizer_params  = normalizer_params or {}
        if weighed_sum:
            gamma = tf.get_variable("gamma", [1], initializer=gamma_initializer)
            out = gamma*out+net
        else:
            out = out+net

        if normalizer_fn is not None:
            out = normalizer_fn(out,**normalizer_params)
        if activation_fn is not None:
            out = activation_fn(out)
        return out

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

def non_local_blockv3(Q,K,V,inner_dims_multiplier=[8,8,2],n_head=1,keep_prob=None,is_training=False,scope=None,
                      conv_op=slim.conv2d,pool_op=None,normalizer_fn=slim.batch_norm,normalizer_params=None,
                      activation_fn=tf.nn.relu,
                      gamma_initializer=tf.constant_initializer(0.0),reuse=None,
                      weighed_sum=True,
                      skip_connect=True):
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

    with tf.variable_scope(scope,default_name="non_local",reuse=reuse):
        shape = wmlt.combined_static_and_dynamic_shape(Q)
        channel = btf.channel(V)
        shape[-1] = channel
        m_channelq = btf.channel(Q)//inner_dims_multiplier[0]
        m_channelk = btf.channel(K)//inner_dims_multiplier[1]
        m_channelv = btf.channel(V)//inner_dims_multiplier[2]
        net = V
        Q = conv_op(Q,m_channelq,[1,1],activation_fn=None,normalizer_fn=None,scope="q_conv")
        K = conv_op(K,m_channelk,[1,1],activation_fn=None,normalizer_fn=None,scope="k_conv")
        V = conv_op(V,m_channelv,[1,1],activation_fn=None,normalizer_fn=None,scope="v_conv")
        if pool_op is not None:
            K = pool_op(K,kernel_size=2, stride=2,padding="SAME")
            V = pool_op(V,kernel_size=2, stride=2,padding="SAME")
        Q = reshape_net(Q)
        K = reshape_net(K)
        V = reshape_net(V)
        out = nlpl.multi_head_attention(Q, K, V, n_head=n_head,keep_prob=keep_prob, is_training=is_training,
                                        use_mask=False)
        out = restore_shape(out,shape,m_channelv)
        out = conv_op(out,channel,[1,1],
                      activation_fn=None,
                      normalizer_fn=None,
                      scope="attn_conv")
        normalizer_params  = normalizer_params or {}
        if skip_connect:
            if weighed_sum:
                gamma = tf.get_variable("gamma", [1], initializer=gamma_initializer)
                out = gamma*out+net
            else:
                out = out + net
        if normalizer_fn is not None:
            out = normalizer_fn(out,**normalizer_params)
        if activation_fn is not None:
            out = activation_fn(out)
        return out


def non_local_blockv4(net,inner_dims_multiplier=[1,1,1],
                      inner_dims=None,
                      n_head=1,keep_prob=None,is_training=False,scope=None,
                      conv_op=slim.conv2d,pool_op=None,normalizer_fn=slim.batch_norm,normalizer_params=None,
                      activation_fn=tf.nn.relu,
                      gamma_initializer=tf.constant_initializer(0.0),reuse=None,
                      weighed_sum=True,pos_embedding=None,
                      size=None):
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
    if inner_dims is not None:
        if isinstance(inner_dims, int):
            inner_dims = [inner_dims]
        if len(inner_dims) == 1:
            inner_dims = inner_dims*3

    with tf.variable_scope(scope,default_name="non_localv4",reuse=reuse):
        org_net = net

        if size is not None:
            net = tf.image.resize_bilinear(net,size)
        shape = wmlt.combined_static_and_dynamic_shape(net)
        with tf.variable_scope("pos_embedding"):
            if pos_embedding is None:
                pos_embs_shape = [1,shape[1],shape[2],shape[3]]
                pos_embedding = tf.get_variable("pos_embs",shape=pos_embs_shape,dtype=tf.float32,
                                                initializer=tf.random_normal_initializer(stddev=0.02))
            net = net+pos_embedding
        channel = shape[-1]
        if inner_dims is not None:
            m_channelq = inner_dims[0]
            m_channelk = inner_dims[1]
            m_channelv = inner_dims[2]
            pass
        else:
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
        out = conv_op(out,channel,[1,1],
                      activation_fn=None,
                      normalizer_fn=None,
                      scope="attn_conv")
        normalizer_params  = normalizer_params or {}
        if size is not None:
            out = tf.image.resize_bilinear(out,wmlt.combined_static_and_dynamic_shape(org_net)[1:3])
        if weighed_sum:
            gamma = tf.get_variable("gamma", [1], initializer=gamma_initializer)
            out = gamma*out+org_net
        else:
            out = out+org_net

        if normalizer_fn is not None:
            out = normalizer_fn(out,**normalizer_params)
        if activation_fn is not None:
            out = activation_fn(out)
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

def cnn_self_attenation(net,channel=None,n_head=1,keep_prob=None,is_training=False,scope=None):
    with tf.variable_scope(scope,"attenation"):
        old_channel = net.get_shape().as_list()[-1]
        if channel is not None:
            net = slim.conv2d(net,channel,[1,1],scope="projection_0")
        shape = wmlt.combined_static_and_dynamic_shape(net)
        new_shape = [shape[0],shape[1]*shape[2],shape[3]]
        net = tf.reshape(net,new_shape)
        net = nlpl.self_attenation(net,n_head=n_head,keep_prob=keep_prob,is_training=is_training,use_mask=False)
        if channel is not None:
            out_shape = [shape[0],shape[1],shape[2],channel]
            net = tf.reshape(net,out_shape)
            net = slim.conv2d(net,old_channel,[1,1],scope="projection_1")
        else:
            out_shape = [shape[0],shape[1],shape[2],old_channel]
            net = tf.reshape(net,out_shape)
        return net

def cnn_self_hattenation(net,channel=None,n_head=1,keep_prob=None,is_training=False,scope=None):
    with tf.variable_scope(scope,"hattenation"):
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

def cnn_self_vattenation(net,channel=None,n_head=1,keep_prob=None,is_training=False,scope=None):
    with tf.variable_scope(scope,"hattenation"):
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

'''
CBAM: Convolutional Block Attention Module
'''
def cbam_block(net,r=16,fc_op=slim.fully_connected,conv_op=slim.conv2d,activation_fn=tf.nn.relu,scope=None,summary_fn=None):
    with tf.variable_scope(scope,"CBAM"):
        with tf.variable_scope("ChannelAttention"):
            channel = net.get_shape().as_list()[-1]
            mid_channel = channel//r
            org_net = net
            net0 = tf.reduce_mean(net,axis=[1,2],keepdims=False)
            net1 = tf.reduce_max(net,axis=[1,2],keepdims=False)
            net = tf.concat([net0,net1],axis=-1)
            net = fc_op(net,mid_channel,activation_fn=activation_fn,normalizer_fn=None)
            net = fc_op(net,channel,activation_fn=tf.nn.sigmoid,normalizer_fn=None)
            if summary_fn is not None:
                summary_fn("se_sigmoid_value",net)
            net = tf.expand_dims(net,axis=1)
            net = tf.expand_dims(net,axis=1)
            net = net*org_net

        with tf.variable_scope("SpatialAttention"):
            net0 = tf.reduce_mean(net,axis=-1,keepdims=True)
            net1 = tf.reduce_max(net,axis=-1,keepdims=True)
            org_net = net
            net = tf.concat([net0,net1],axis=-1)
            net = conv_op(net,1,[7,7],activation_fn=tf.nn.sigmoid,normalizer_fn=None)
            net = net*org_net

        return net

def get_dropblock_keep_prob(step,total_step,max_keep_prob):
    return 1.0-tf.minimum(tf.cast(step,tf.float32)/total_step,1.0)*(1-max_keep_prob)

def dropblock(inputs,keep_prob,is_training,block_size=7,scope=None,seed=int(time.time()),all_channel=False):
    with tf.variable_scope(scope,default_name="dropblock"):
        if not is_training:
            return tf.identity(inputs)
        _,width,height,_ = btf.combined_static_and_dynamic_shape(inputs)
        drop_prob = (1.0 - keep_prob) * tf.to_float(width * height) / block_size ** 2 / tf.to_float((
                width - block_size + 1)*(height-block_size+1))
        h_i,w_i = tf.meshgrid(tf.range(height), tf.range(width))
        valid_block_center = tf.logical_and(
            tf.logical_and(w_i >= int(block_size // 2),
                           w_i < width - (block_size - 1) // 2),
            tf.logical_and(h_i >= int(block_size // 2),
                           h_i < height- (block_size - 1) // 2))
        valid_block_center = tf.expand_dims(valid_block_center, 0)
        valid_block_center = tf.expand_dims(valid_block_center, -1)
        if all_channel:
            mask = tf.random_uniform(shape=tf.shape(inputs)[:-1],minval=0.,maxval=1.0,dtype=tf.float32,seed=seed)
            mask = tf.expand_dims(mask,axis=-1)
        else:
            mask = tf.random_uniform(shape=tf.shape(inputs),minval=0.,maxval=1.0,dtype=tf.float32,seed=seed)
        mask = mask+(1.0-tf.to_float(valid_block_center))
        bin_mask = tf.greater(mask,drop_prob)
        bin_mask = tf.cast(bin_mask,tf.float32)
        bin_mask = -tf.nn.max_pool(-bin_mask,ksize=[1,block_size,block_size,1],strides=[1,1,1,1],padding="SAME")
        nozero = tf.reduce_sum(bin_mask)
        allsize = tf.cast(tf.reduce_prod(tf.shape(bin_mask)),tf.float32)
        ratio = allsize/nozero
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
            if w.dtype == tf.float16:
                w = tf.cast(w,tf.float32)
            c = w.get_shape().as_list()[-1]

            w = tf.reshape(w, [-1, c])

            """ Declaring a Identity Tensor of appropriate size"""
            identity = tf.eye(c,dtype=tf.float32)

            """ Regularizer Wt*W(1 - I) """
            w_mul = tf.matmul(w, w,transpose_a=True)
            mask = tf.ones_like(identity)-identity
            reg = mask*w_mul

            """Calculating the Loss Obtained"""
            ortho_loss = tf.nn.l2_loss(reg)

            return scale * ortho_loss

    return ortho_reg


def scale_gradient(x,scale,is_training=True):
    if not is_training:
        return x
    if not isinstance(scale,tf.Tensor):
        scale = tf.constant(scale,tf.float32)
    return __scale_gradient(x,scale)
@tf.custom_gradient
def __scale_gradient(x,scale):
    def grad(dy0):
        return dy0*(tf.ones_like(x)*scale),None
        #return tf.zeros_like(x)
    return (tf.identity(x)),grad

@tf.custom_gradient
def __quantized(res,x):
    with tf.control_dependencies([x]):
        res = tf.identity(res)
    def grad(dy):
        return None,dy
    return res,grad

def quantized_layer(x,e,e_regularizer=orthogonal_regularizerv2(),scale_e=1.0,scale_x=1.0):
    org_data = x
    x = tf.expand_dims(x,axis=-2)
    d = tf.norm(x-e,axis=-1)
    k = tf.argmin(d, axis=-1)
    res = tf.gather(e, k)
    res = __quantized(res,org_data)
    if e_regularizer is not None:
        e_r = e_regularizer(tf.transpose(e))
    l_e = tf.reduce_mean(tf.norm(tf.stop_gradient(res)-x))*scale_e
    l_x = tf.reduce_mean(tf.norm(tf.stop_gradient(x)-res))*scale_x
    loss = {}
    loss["l_e"] = l_e
    loss["l_x"] = l_x
    loss["e_r"] = e_r
    return res,loss

@add_arg_scope
def deform_conv2d(inputs,
                  offset,
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
                  num_groups=1,
                  deformable_group=1,
                  reuse=None,
                  scope=None):
    '''
    :param inputs: [B,H,W,C] or [B,C,H,W]
    :param offset: [B,H,W,num_groups*kernel_size[0]*kernel_size[1]*2]
    :param num_outputs:
    :param kernel_size:
    :param stride:
    :param padding:
    :param activation_fn:
    :param weights_initializer:
    :param weights_regularizer:
    :param biases_initializer:
    :param biases_regularizer:
    :param normalizer_fn:
    :param normalizer_params:
    :param outputs_collections:
    :param rate:
    :param reuse:
    :param scope:
    :return:
    '''
    with variable_scope.variable_scope(scope, 'deform_conv2d', [inputs], reuse=reuse) as sc:
        in_channels = inputs.get_shape().as_list()[-1]
        assert num_outputs%num_groups==0,"error num outputs."
        if isinstance(kernel_size,list):
            shape = [num_outputs,in_channels//num_groups]+kernel_size
        else:
            shape = [num_outputs,in_channels//num_groups,kernel_size,kernel_size]
        w = tf.get_variable("kernel", shape=shape,
                            initializer=weights_initializer)
        b = tf.get_variable("bias", [num_outputs], initializer=biases_initializer)
        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(w))
        if biases_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,biases_regularizer(b))

        inputs = tf.transpose(inputs,perm=[0,3,1,2])
        offset = tf.transpose(offset,perm=[0,3,1,2])
        outputs = wtfop.deform_conv_op(x=inputs,
                                       filter=w,
                                       offset=offset,
                                       strides=[1, 1,stride, stride],padding=padding,
                                       rates=[1,1,rate,rate],
                                       num_groups=num_groups,
                                       deformable_group=deformable_group,
                                       data_format="NCHW")
        outputs = tf.transpose(outputs,perm=[0,2,3,1]) + b
        if normalizer_fn is not None:
            if normalizer_params is None:
                normalizer_params = {}
            outputs = normalizer_fn(outputs,**normalizer_params)
        if activation_fn is not None:
            outputs = utils.collect_named_outputs(outputs_collections, sc.name+"_pre_act", outputs)
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)
'''
@add_arg_scope
def deform_conv2dv2(inputs,
                  num_outputs,
                  kernel_size,
                  deformable_group=1,
                  scope=None,
                  **kwargs
                  ):
    with tf.variable_scope(scope,"deform_conv2d"):
        if not isinstance(kernel_size,Iterable):
            kernel_size = [kernel_size,kernel_size]
        offset = slim.conv2d(inputs,deformable_group*2*kernel_size[0]*kernel_size[1],[1,1],scope="get_offset",
                             activation_fn=None,
                             normalizer_fn=None)
        return deform_conv2d(inputs=inputs,
                             offset=offset,
                             kernel_size=kernel_size,
                             num_outputs=num_outputs,
                             deformable_group=deformable_group,**kwargs)
'''


def swish(features):
  # pylint: disable=g-doc-args
  """Computes the Swish activation function: `x * sigmoid(x)`.

  Source: "Searching for Activation Functions" (Ramachandran et al. 2017)
  https://arxiv.org/abs/1710.05941

  Args:
    features: A `Tensor` representing preactivation values.
    name: A name for the operation (optional).

  Returns:
    The activation value.
  """
  features = ops.convert_to_tensor(features, name="features")
  return features * math_ops.sigmoid(features)

# Minibatch standard deviation layer.

@add_arg_scope
def minibatch_stddev_layer(x, group_size=4, num_new_features=1,scope=None):
    with tf.name_scope(scope,"minibatch_stddev_layer"):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = wmlt.combined_static_and_dynamic_shape(x)
        y = tf.reshape(x, [group_size, -1, s[1],s[2],num_new_features, s[3]//num_new_features])   # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y, axis=[1,2,4], keepdims=True)      # [Mn111]  Take average over fmaps and pixels.
        y = tf.squeeze(y,axis=4)
        y = tf.tile(y, [group_size,s[1], s[2],1])             # [NnHW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=3)                        # [NCHW]  Append as new fmap.

@add_arg_scope
def group_conv2d(inputs,
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
                 weights_normalizer_fn=None,
                 weights_normalizer_params=None,
                 outputs_collections=None,
                 groups=32,
                 rate=1,
                 reuse=None,
                 scope=None):

    with variable_scope.variable_scope(scope, 'group_conv2d', [inputs], reuse=reuse) as sc:
        B,H,W,C = wmlt.combined_static_and_dynamic_shape(inputs)
        if not isinstance(kernel_size,Iterable):
            kernel_size = [kernel_size,kernel_size]
        assert C%groups==0,f"Input num must exactly divisible ty groups"
        shape = [groups]+list(kernel_size)+[C//groups,num_outputs//groups]
        w = tf.get_variable("kernel", shape=shape,
                            initializer=weights_initializer)
        if normalizer_fn is None:
            b = tf.get_variable("bias", [num_outputs], initializer=biases_initializer)
        else:
            b = None

        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(w))

        if b is not None and biases_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,biases_regularizer(b))

        outputs = []
        inputs = tf.split(inputs,groups,axis=-1)

        if not isinstance(rate,Iterable):
            rate = [rate,rate]

        for i in range(groups):
            wi = w[i]
            if weights_normalizer_fn is not None:
                if weights_normalizer_params is None:
                    weights_normalizer_params = {}
                wi = weights_normalizer_fn(wi,**weights_normalizer_params)
            net = tf.nn.conv2d(input=inputs[i], filter=wi,
                                   strides=[1, stride, stride, 1], padding=padding,dilations=[1]+rate+[1])
            outputs.append(net)

        outputs = tf.concat(outputs,axis=-1)

        if b is not None:
            outputs = outputs+b

        if normalizer_fn is not None:
            if normalizer_params is None:
                normalizer_params = {}
            outputs = normalizer_fn(outputs,**normalizer_params)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

@add_arg_scope
def group_conv2d_with_sn(inputs,
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
                 groups=32,
                 rate=1,
                 reuse=None,
                 scope=None):

    with variable_scope.variable_scope(scope, 'group_conv2d', [inputs], reuse=reuse) as sc:
        B,H,W,C = wmlt.combined_static_and_dynamic_shape(inputs)
        if not isinstance(kernel_size,Iterable):
            kernel_size = [kernel_size,kernel_size]
        assert C%groups==0,f"Input num must exactly divisible ty groups"
        shape = [groups]+list(kernel_size)+[C//groups,num_outputs//groups]
        w = tf.get_variable("kernel", shape=shape,
                            initializer=weights_initializer)
        if normalizer_fn is None:
            b = tf.get_variable("bias", [num_outputs], initializer=biases_initializer)
        else:
            b = None

        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(w))

        if b is not None and biases_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,biases_regularizer(b))

        outputs = []
        inputs = tf.split(inputs,groups,axis=-1)

        if not isinstance(rate,Iterable):
            rate = [rate,rate]

        for i in range(groups):
            net = tf.nn.conv2d(input=inputs[i], filter=spectral_norm(w[i],scope=f"spectral_norm{i}"),
                               strides=[1, stride, stride, 1], padding=padding,dilations=[1]+rate+[1])
            outputs.append(net)

        outputs = tf.concat(outputs,axis=-1)

        if b is not None:
            outputs = outputs+b

        if normalizer_fn is not None:
            if normalizer_params is None:
                normalizer_params = {}
            outputs = normalizer_fn(outputs,**normalizer_params)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

@add_arg_scope
def conv2d(inputs,
           num_outputs,
           kernel_size,
           stride=1,
           padding='SAME',
           activation_fn=nn.relu,
           weights_initializer=initializers.xavier_initializer(),
           weights_regularizer=None,
           biases_initializer=init_ops.zeros_initializer(),
           biases_regularizer=None,
           weights_normalizer_fn=None,
           weights_normalizer_params=None,
           normalizer_fn=None,
           normalizer_params=None,
           outputs_collections=None,
           rate=1,
           reuse=None,
           scope=None):
    del rate
    with variable_scope.variable_scope(scope, 'conv2d', [inputs], reuse=reuse) as sc:
        if isinstance(kernel_size,list):
            shape = kernel_size+[inputs.get_shape().as_list()[-1],num_outputs]
        else:
            shape = [kernel_size,kernel_size,inputs.get_shape().as_list()[-1],num_outputs]
        w = tf.get_variable("kernel", shape=shape,
                            initializer=weights_initializer)
        if biases_initializer is not None and normalizer_fn is None:
            b = tf.get_variable("bias", [num_outputs], initializer=biases_initializer)
        else:
            b = None
        if weights_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,weights_regularizer(w))
        if b is not None and biases_regularizer is not None:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,biases_regularizer(b))

        if weights_normalizer_fn is not None:
            if weights_normalizer_params is None:
                weights_normalizer_params = {}
            w = weights_normalizer_fn(w,**weights_normalizer_params)

        outputs = tf.nn.conv2d(input=inputs, filter=w, strides=[1, stride, stride, 1],padding=padding)
        if b is not None:
            outputs = outputs + b
        if normalizer_fn is not None:
            if normalizer_params is None:
                normalizer_params = {}
            outputs = normalizer_fn(outputs,**normalizer_params)
        if activation_fn is not None:
            outputs = utils.collect_named_outputs(outputs_collections, sc.name+"_pre_act", outputs)
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

def upsample(x,scale_factor=2,mode=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    if len(x.get_shape())==4:
        B,H,W,_ = btf.combined_static_and_dynamic_shape(x)
    elif len(x.get_shape())==3:
        H, W, _ = btf.combined_static_and_dynamic_shape(x)
    else:
        raise NotImplementedError("Error")

    if isinstance(scale_factor,(int,float)):
        scale_factor = [scale_factor,scale_factor]

    H = tf.to_float(H)
    W = tf.to_float(W)
    return tf.image.resize_images(x,[tf.to_int32(H*scale_factor[0]+0.1),tf.to_int32(W*scale_factor[1]+0.1)],method=mode)
