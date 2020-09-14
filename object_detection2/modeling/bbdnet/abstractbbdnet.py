import numpy as np
import tensorflow as tf
from GN.graph import DynamicAdjacentMatrix
import object_detection.bboxes as odb
from wtfop.wtfop_ops import adjacent_matrix_generator_by_iou
import wnnlayer as wnnl
import wml_tfutils as wmlt
from functools import partial
import wnn
slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

EDGE_HIDDEN_SIZE=128
POINT_HIDDEN_SIZE=128
GLOBAL_HIDDEN_SIZE=128

'''
Each time only process one example.
'''
class AbstractBBDNet:
    '''
    boxes: the  boxes, [batch_size=1,k,4]
    probability: [batch_size=1,k,classes_num] the probability of boxes
    map_data:[batch_size=1,k,X,X,C]
    classes_num: ...
    '''
    def __init__(self,cfg,boxes,probability,map_data,classes_num,base_net,is_training=False):
        assert boxes.get_shape().ndims==2, "error"
        assert probability.get_shape().ndims==2, "error"
        assert map_data.get_shape().ndims==2 or map_data.get_shape().ndims==4, "error"

        self.cfg = cfg
        self.logits = None
        #edges between guidepost
        self.boxes = boxes
        self.probability = probability
        self.map_data = map_data
        self.classes_num = classes_num

        boxes_shape = boxes.get_shape().as_list()
        self.p_nr = boxes_shape[1]

        self.A = None
        self.Status = None
        self.is_training = is_training
        self.base_net = base_net
        self.mid_outputs = []

    def get_predict(self,proposal_boxes,threshold=None):
        probs = tf.nn.softmax(self.logits)
        if threshold is not None:
            probs = probs[:,:,1:]
        probs,raw_labels= tf.nn.top_k(probs,k=1)
        if threshold is not None:
            raw_labels = raw_labels+1
        raw_labels = tf.reshape(raw_labels,[-1])
        probs = tf.reshape(probs,[-1])
        mask = tf.greater(raw_labels,0)
        if threshold is not None:
            mask = tf.logical_and(mask,tf.greater(probs,threshold))
        boxes = tf.boolean_mask(proposal_boxes,mask)
        labels = tf.boolean_mask(raw_labels,mask)
        probs = tf.boolean_mask(probs,mask)
        return boxes,labels,probs,raw_labels

    '''
    y:[batch_size,k] target label
    '''
    def loss(self,y):
        assert y.get_shape().ndims==1, "error"
        loss = []
        with tf.variable_scope("losses"):
            for i,logits in enumerate(self.mid_outputs):
                p_loss,n_loss = AbstractBBDNet._loss(logits,y)
                #scale = (float(i)+1.0)/5.0
                scale = 1.0
                loss.append(p_loss*scale)
                loss.append(n_loss*scale)

        return tf.add_n(loss)

    def auxiliary_loss(self,y):
        assert y.get_shape().ndims==2, "error"
        with tf.variable_scope("AuxiliaryLoss"):
            def output_fn(net):
                with tf.variable_scope("Output",reuse=tf.AUTO_REUSE):
                    net = slim.fully_connected(net,64)
                    net = slim.fully_connected(net,self.classes_num+1,activation_fn=None)
                    return net
            input_net = self.mid_outputs[0]
            net = output_fn(input_net)
            p_loss,n_loss = AbstractBBDNet._loss(net,y)
            tf.losses.add_loss(p_loss)
            tf.losses.add_loss(n_loss)

    @staticmethod
    def _loss(logits,y):
        assert y.get_shape().ndims==1, "error"
        assert logits.get_shape().ndims==3, "error"
        #p_mask = tf.greater(y,0)
        #logits = tf.Print(logits,["shape logits",tf.shape(logits)])
        with tf.device(":/cpu:0"):
            logits = tf.squeeze(logits,axis=0)
            values,indices = tf.nn.top_k(logits)
            plabels = tf.reshape(indices,[-1])
            p_mask = tf.logical_or(tf.greater(y,0),tf.greater(plabels,0))
            n_mask = tf.logical_not(p_mask)
            p_y = tf.boolean_mask(y,p_mask)
            p_logits = tf.boolean_mask(logits,p_mask)
            n_y = tf.boolean_mask(y,n_mask)
            n_logits = tf.boolean_mask(logits,n_mask)
            p_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p_logits,labels=p_y)
            n_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=n_logits,labels=n_y)
            p_size = tf.reduce_sum(tf.cast(p_mask,tf.int32))
            n_size = tf.reduce_sum(tf.cast(n_mask,tf.int32))
            p_loss = tf.cond(p_size>0,lambda:tf.reduce_mean(p_loss),lambda :tf.constant(0,dtype=tf.float32))
            n_loss = tf.cond(n_size>0,lambda:tf.reduce_mean(n_loss),lambda :tf.constant(0,dtype=tf.float32))
            #loss = wnn.sparse_softmax_cross_entropy_with_logits_FL(logits=self.logits,labels=y)
            return p_loss,n_loss

    @staticmethod
    def linear(x,dims,scope=None):
        with tf.variable_scope(scope,default_name="MLP",reuse=tf.AUTO_REUSE):
            net = slim.fully_connected(x, dims,activation_fn=None)
            return net

    @staticmethod
    def _mlp(x,dims,scope=None,pool=False):
        with tf.variable_scope(scope,default_name="MLP",reuse=tf.AUTO_REUSE):
            mid_dims = dims if not pool else 2*dims
            net = slim.fully_connected(x, mid_dims)
            if pool:
                x = tf.expand_dims(net,axis=-1)
                x = tf.layers.max_pooling1d(x,strides=2,pool_size=2)
                net = tf.squeeze(x,axis=-1)
            return net

    @staticmethod
    def res_mlp(x,dims,scope=None,pool=False):
        with tf.variable_scope(scope,default_name="bottleneck_v1",reuse=tf.AUTO_REUSE):
            in_dims = x.get_shape().as_list()[-1]
            mid_dims = dims if not pool else 2*dims
            half_mid_dims = mid_dims//2
            net = x
            #net = wmlt.PrintNaNorInf(net,name="is_nan_inf_in_mlp")
            if in_dims != mid_dims:
                x = slim.fully_connected(x, mid_dims,activation_fn=None,
                                         normalizer_fn=None,
                                         scope = "shortcut")
            net = slim.fully_connected(net, half_mid_dims)
            net = slim.fully_connected(net, half_mid_dims)
            net = slim.fully_connected(net, mid_dims)
            if pool:
                net = AbstractBBDNet.max_pool(net,strides=2,pool_size=2)
            return net+x

    @staticmethod
    def res_block(x,dims,unit_nr=3,scope=None,pool=False):
        with tf.variable_scope(scope,default_name="block",reuse=tf.AUTO_REUSE):
            for i in range(unit_nr):
                is_pool = pool and (i==unit_nr-1)
                x = AbstractBBDNet.res_mlp(x,dims,f"unit_{i+1}",is_pool)
            return x

    @staticmethod
    def max_pool(net,strides=2,pool_size=2):
        net = tf.expand_dims(net,axis=-1)
        net = tf.layers.max_pooling1d(net,strides=strides,pool_size=pool_size)
        net = tf.squeeze(net,axis=-1)
        return net

    @staticmethod
    def mlp(x,dims,scope=None,layer=4,pool_last=False):
        with tf.variable_scope(scope,default_name="MLP"):
            mid_dims = dims if not pool_last else 2*dims
            for i in range(layer):
                x = AbstractBBDNet._mlp(x,mid_dims,scope=f"SubLayer{i}")
            if pool_last:
                x = tf.expand_dims(x,axis=-1)
                x = tf.layers.max_pooling1d(x,strides=2,pool_size=2)
                x = tf.squeeze(x,axis=-1)
        return x