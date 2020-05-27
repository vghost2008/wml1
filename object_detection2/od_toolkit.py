#coding=utf-8
import tensorflow as tf
import wtfop.wtfop_ops as wop
import wml_tfutils as wmlt
import wnnlayer as wnnl

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
    if name in activation_dict:
        return activation_dict[name]
    else:
        raise KeyError(f"Can't find activation fn {name}.")

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


