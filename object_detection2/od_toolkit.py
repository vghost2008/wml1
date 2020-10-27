#coding=utf-8
import tensorflow as tf
import wtfop.wtfop_ops as wop
import wml_tfutils as wmlt
import wnnlayer as wnnl
import basic_tftools as btf
from .standard_names import *

slim = tf.contrib.slim

'''
random select size boxes from the input boxes
boxes:[N,4]
len:() indict the size of boxes, or None
return:
[size,4]
'''
def random_select_boxes(boxes,size,len=None):
    with tf.name_scope("random_select_boxes"):
        if len is None:
            data_nr = tf.shape(boxes)[0]
        else:
            data_nr = len
        indexs = tf.range(data_nr)
        indexs = wop.wpad(indexs, [0, tf.reshape(size - data_nr,())])
        indexs = tf.random_shuffle(indexs)
        indexs = tf.random_crop(indexs, [size])
        boxes = tf.gather(boxes, indexs)
        return boxes, indexs

'''
boxes:[batch_size,N,4]
lens:[batch_size]
return:
[batch_size,size,4]
'''
def batched_random_select_boxes(boxes,lens,size):
    with tf.name_scope("random_select_boxes"):
        boxes,indexs = tf.map_fn(lambda x:random_select_boxes(x[0],size,x[1]),elems=(boxes,lens),dtype=(tf.float32,tf.int32))
        batch_size = boxes.get_shape().as_list()[0]
        boxes = wmlt.reshape(boxes,[batch_size,size,4])
        indexs = wmlt.reshape(indexs,[batch_size,size])
        return boxes,indexs

def get_norm_scope_name(name:str):
    scope_name_dict = {'BN':'BatchNorm',
                       'GN':'GroupNorm',
                       'evo_norm_s0':'EvoNormS0',
                       'SN':'SpectralNorm',
                       'NA':None}
    if name in scope_name_dict:
        return scope_name_dict[name]

    if len(name) == 0:
        return None
    return "Norm"

def get_norm(name:str,is_training):
    if len(name) == 0:
        return None,None,
    elif name == "BN":
        norm_params = {
            'decay': 0.997,
            'epsilon': 1e-4,
            'scale': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'fused': None,  # Use fused batch norm if possible.
            'is_training': is_training
        }
        return slim.batch_norm,norm_params
    elif name == "GN":
        norm_params = {
            "G":32,
            "epsilon":1e-5,
            "weights_regularizer":None,
            "scale":True,
            "offset":True,
        }
        return wnnl.group_norm,norm_params
    elif name == "evo_norm_s0":
        norm_params = {
            "G": 32,
            "epsilon": 1e-5,
            "weights_regularizer": None,
            "scale": True,
        }
        return wnnl.evo_norm_s0, norm_params
    elif name == "GNV0":
        norm_params = {
            "G": 32,
            "epsilon": 1e-5,
            "weights_regularizer": None,
            "scale": True,
            "offset": True,
            "scope":"group_norm"
        }
        return wnnl.group_norm_4d_v0, norm_params
    elif name == "SN":
        norm_params = {
            'is_training': is_training
        }
        return wnnl.spectral_norm_for_conv, norm_params
    else:
        raise NotImplementedError()
def get_activation_fn(name):
    activation_dict = {"relu":tf.nn.relu,
                       "relu6":tf.nn.relu6,
                       "NA":None,
                       "swish":tf.nn.swish,
                       "sigmoid":tf.nn.sigmoid,
                       "tanh":tf.nn.tanh}
    if len(name) == 0:
        return None
    if name in activation_dict:
        return activation_dict[name]
    else:
        raise KeyError(f"Can't find activation fn {name}.")

def get_fusion_fn(name):
    fusion_dict = {"concat":concat_fusion,
                   "sum":sum_fusion,
                   "avg":avg_fusion,
                   "mix_fusion":mix_fusion,
                   "wsum":fusion}
    if len(name) == 0:
        return sum_fusion
    if name in fusion_dict:
        return fusion_dict[name]
    else:
        raise KeyError(f"Can't find fusion fn {name}.")

def fusion(feature_maps,depth=None,scope=None):
    with tf.variable_scope(scope,"fusion"):
        if depth is None:
            depth = feature_maps[-1].get_shape().as_list()[-1]
        out_feature_maps = []
        ws = []
        for i,net in enumerate(feature_maps):
            wi = tf.get_variable(f"w{i}",shape=(),dtype=tf.float32,initializer=tf.ones_initializer)
            wi = tf.nn.relu(wi)
            ws.append(wi)
        sum_w = tf.add_n(ws)+1e-4
        for i,net in enumerate(feature_maps):
            if net.get_shape().as_list()[-1] != depth:
                raise RuntimeError("Depth must be equal.")
            out_feature_maps.append(net*ws[i]/sum_w)
        return tf.add_n(out_feature_maps)

def sum_fusion(feature_maps,*args,scope=None,**kwargs):
    if len(feature_maps) == 2:
        return tf.add(feature_maps[0],feature_maps[1],name=scope)
    else:
        return tf.add_n(feature_maps,name=scope)

def avg_fusion(feature_maps,*args,scope=None,**kwargs):
    if len(feature_maps) == 2:
        return tf.add(feature_maps[0],feature_maps[1],name=scope)/len(feature_maps)
    else:
        return tf.add_n(feature_maps,name=scope)/len(feature_maps)

def concat_fusion(feature_maps,*args,scope=None,**kwargs):
    return tf.concat(feature_maps,axis=-1,name=scope)

def mix_fusion(feature_maps,*args,concat_nr=32,scope=None,**kwargs):
    concat_feature_maps = []
    sum_feature_maps = []
    with tf.name_scope(scope,default_name="mix_fusion"):
        concat_feature_maps.append(feature_maps[-1][..., :concat_nr])
        for net in feature_maps:
            sum_feature_maps.append(net[...,concat_nr:])
        if len(feature_maps) == 2:
            s_net = tf.add(sum_feature_maps[0],sum_feature_maps[1])
        else:
            s_net = tf.add_n(sum_feature_maps)

        concat_feature_maps.append(s_net)

        return tf.concat(concat_feature_maps,axis=-1)

def boolean_mask_on_instances(instances,mask,labels_key=RD_LABELS,length_key=RD_LENGTH,exclude=[IMAGE]):
    res = {}
    B,size = btf.combined_static_and_dynamic_shape(instances[labels_key])
    for k, v in instances.items():
        if len(v.get_shape())>1 and k not in exclude:
            n_v = wmlt.batch_boolean_mask(v, mask, size=size,scope=k)
            res[k] = n_v
        else:
            res[k] = v
    length = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
    res[length_key] = length

    return res
