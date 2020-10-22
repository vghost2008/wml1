import numpy as np
import tensorflow as tf
from GN.graph import DynamicAdjacentMatrix
import wsummary
import object_detection.bboxes as odb
from wtfop.wtfop_ops import adjacent_matrix_generator_by_iou
from object_detection2.modeling.box_regression import Box2BoxTransform
import wnnlayer as wnnl
import wml_tfutils as wmlt
from functools import partial
import basic_tftools as btf
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
    def __init__(self,cfg,boxes,map_data,is_training=False):
        assert boxes.get_shape().ndims==2, "error"
        assert map_data.get_shape().ndims==2 or map_data.get_shape().ndims==4, "error"

        self.cfg = cfg
        self.logits = None
        self.pred_bboxes_deltas = None
        #edges between guidepost
        self.boxes = boxes
        self.map_data = map_data

        boxes_shape = boxes.get_shape().as_list()
        self.p_nr = boxes_shape[1]

        self.A = None
        self.Status = None
        self.is_training = is_training
        self.mid_outputs = []
        self.mid_bboxes_outputs = []
        self.box2box_transform = Box2BoxTransform()


    def get_nms_predict(self, threshold=0.5):
        '''
        :param proposal_boxes:
        :param threshold:
        nms_logits: [N]
        :return:
        '''
        probs = tf.nn.sigmoid(self.nms_logits)
        mask = tf.greater(probs, threshold)
        return probs,mask

    def get_predict(self, proposal_boxes, raw_labels,raw_probs,threshold=None):
        raw_labels = tf.reshape(raw_labels, [-1])
        probs = tf.reshape(raw_probs, [-1])
        nms_probs,nms_mask = self.get_nms_predict(threshold[1])
        mask =  nms_mask
        boxes = tf.boolean_mask(proposal_boxes, mask)
        pred_deltas = tf.boolean_mask(self.pred_bboxes_deltas,mask)
        labels = tf.boolean_mask(raw_labels, mask)
        probs = tf.boolean_mask(probs, mask)
        boxes = self.box2box_transform.apply_deltas(pred_deltas,boxes)
        return boxes, labels, probs, raw_labels

    def get_predictv2(self, proposal_boxes, raw_labels,raw_probs,threshold=None):
        edges_send_node, edges_receive_node= self.A.senders_indexs, self.A.receivers_indexs
        edges_logits = self.mid_edges_outputs[-1]
        raw_labels = tf.reshape(raw_labels, [-1])
        probs = tf.reshape(raw_probs, [-1])
        nms_probs = tf.clip_by_value(tf.nn.sigmoid(self.nms_logits),0,0.5)*2
        edges_probs = tf.nn.sigmoid(edges_logits)
        #probs = probs*nms_probs
        mask = tf.py_func(self.get_predict_by_edges,inp=(probs,edges_send_node,edges_receive_node,edges_probs,threshold[1]),
                          Tout=tf.bool)
        shape = wmlt.combined_static_and_dynamic_shape(probs)
        mask = tf.reshape(mask,shape)
        boxes = tf.boolean_mask(proposal_boxes, mask)
        pred_deltas = tf.boolean_mask(self.pred_bboxes_deltas,mask)
        labels = tf.boolean_mask(raw_labels, mask)
        probs = tf.boolean_mask(probs, mask)
        boxes = self.box2box_transform.apply_deltas(pred_deltas,boxes)
        return boxes, labels, probs, raw_labels

    @staticmethod
    def get_predict_by_edges(probs,edges_send_node,edges_receive_node,edges_probs,edges_threshold):
        node_nr = probs.shape[0]
        edges_nr = edges_send_node.shape[0]
        mask = np.ones([node_nr],dtype=np.bool)
        processed_node = set()
        for i in range(edges_nr):
            s_n = edges_send_node[i]
            r_n = edges_receive_node[i]
            if s_n in processed_node and r_n in processed_node:
                continue
            processed_node.add(s_n)
            processed_node.add(r_n)
            if edges_probs[i]<edges_threshold:
                continue
            need_dd_node = set()
            need_dd_node.add(s_n)
            need_dd_node.add(r_n)

            for j in range(i+1,edges_nr):
                c_s_n = edges_send_node[j]
                c_r_n = edges_receive_node[j]
                if c_s_n in processed_node and c_r_n in processed_node:
                    continue
                if edges_probs[j] < edges_threshold:
                    continue
                if c_s_n not in need_dd_node and c_r_n not in need_dd_node:
                    continue
                need_dd_node.add(c_s_n)
                need_dd_node.add(c_r_n)
                processed_node.add(c_s_n)
                processed_node.add(c_r_n)

            max_probs = -1.0
            index = -1
            for v in need_dd_node:
                if probs[v]>max_probs:
                    index = v
                    max_probs = probs[v]
            for v in need_dd_node:
                if v!=index:
                    mask[v] = False
        return mask


    def _lossv1(self,logits,y,log=False):
        assert y.get_shape().ndims==1, "error"
        assert logits.get_shape().ndims==1, "error"
        loss0 = wnn.sigmoid_cross_entropy_with_logits_FL(labels=tf.cast(y,tf.float32),
                                                         logits=logits,
                                                         alpha=None)
        return tf.reduce_mean(loss0)

    def _lossv2(self,logits,y,log=False):
        assert y.get_shape().ndims==1, "error"
        assert logits.get_shape().ndims==1, "error"
        loss0 = wnn.sigmoid_cross_entropy_with_logits_FL(labels=tf.cast(y,tf.float32),
                                                        logits=logits,
                                                         alpha=None)
        plabels = tf.cast(tf.greater(tf.nn.sigmoid(logits),0.5),tf.int32)
        pmask = tf.logical_or(tf.greater(y, 0), tf.greater(plabels, 0))
        nmask = tf.logical_not(pmask)
        p_nr = tf.reduce_sum(tf.cast(pmask,tf.float32))+1
        n_nr = tf.reduce_sum(tf.cast(nmask,tf.float32))+1
        p_scale = tf.clip_by_value(n_nr/p_nr,0.05,20.0)
        ploss = tf.reduce_sum(tf.boolean_mask(loss0, pmask))*p_scale
        nloss = tf.reduce_sum(tf.boolean_mask(loss0, nmask))
        total_nr = tf.stop_gradient(p_nr*p_scale+n_nr)
        tf.summary.scalar("p_scale",p_scale)
        if log:
            ploss = tf.Print(ploss, [ploss, nloss,pmask,nmask,y,logits], summarize=1000)
        return (ploss+nloss)/total_nr

    def _lossv3(self,logits,y,log=False):
        assert y.get_shape().ndims==1, "error"
        assert logits.get_shape().ndims==1, "error"
        loss0 = wnn.sigmoid_cross_entropy_with_logits_FL(labels=tf.cast(y,tf.float32),
                                                        logits=logits,
                                                         alpha=None)
        pmask = tf.greater(y, 0)
        nmask = tf.logical_not(pmask)
        p_nr = tf.reduce_sum(tf.cast(pmask,tf.float32))+1
        n_nr = tf.reduce_sum(tf.cast(nmask,tf.float32))+1
        n_scale = tf.clip_by_value(p_nr/n_nr,0.05,20.0)
        ploss = tf.reduce_sum(tf.boolean_mask(loss0, pmask))
        nloss = tf.reduce_sum(tf.boolean_mask(loss0, nmask))*n_scale
        tf.summary.scalar("n_scale",n_scale)
        total_nr = tf.stop_gradient(p_nr+n_nr*n_scale)
        if log:
            ploss = tf.Print(ploss, [ploss, nloss,pmask,nmask,y,logits], summarize=1000)
        return (ploss+nloss)/total_nr

    @staticmethod
    def linear(x,dims,scope=None):
        with tf.variable_scope(scope,default_name="MLP",reuse=tf.AUTO_REUSE):
            net = slim.fully_connected(x, dims,activation_fn=None)
            return net

    def _mlp(self,x,dims,scope=None,pool=False):
        with tf.variable_scope(scope,default_name="MLP",reuse=tf.AUTO_REUSE):
            mid_dims = dims if not pool else 2*dims
            net = slim.fully_connected(x, mid_dims,normalizer_fn=self.normalizer_fn)
            if pool:
                x = tf.expand_dims(net,axis=-1)
                x = tf.layers.max_pooling1d(x,strides=2,pool_size=2)
                net = tf.squeeze(x,axis=-1)
            return net

    def res_mlp(self,x,dims,scope=None):
        with tf.variable_scope(scope,default_name="bottleneck_v1",reuse=tf.AUTO_REUSE):
            in_dims = x.get_shape().as_list()[-1]
            net = x
            if in_dims != dims:
                x = slim.fully_connected(x, dims,activation_fn=None,
                                         normalizer_fn=None,
                                         scope = "shortcut")
            normalizer_fn = self.normalizer_fn
            net = slim.fully_connected(net, dims,normalizer_fn=normalizer_fn)
            net = slim.fully_connected(net, dims,normalizer_fn=None,activation_fn=None)
            return tf.nn.leaky_relu(normalizer_fn(net+x))

    def res_block(self,x,dims,unit_nr=1,scope=None):
        with tf.variable_scope(scope,default_name="block",reuse=tf.AUTO_REUSE):
            for i in range(unit_nr):
                x = self.res_mlp(x,dims,f"unit_{i+1}")
            return x

    def mlp(self,x,dims,scope=None,layer=4,pool_last=False):
        with tf.variable_scope(scope,default_name="MLP"):
            mid_dims = dims if not pool_last else 2*dims
            for i in range(layer):
                x = self._mlp(x,mid_dims,scope=f"SubLayer{i}")
            if pool_last:
                x = tf.expand_dims(x,axis=-1)
                x = tf.layers.max_pooling1d(x,strides=2,pool_size=2)
                x = tf.squeeze(x,axis=-1)
        return x
