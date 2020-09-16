#coding=utf-8
from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
import wml_tfutils as wmlt
import functools
import wtfop.wtfop_ops as wop

class DynamicAdjacentMatrix:
    '''
    adj_mt: used to indict connection, row i and col j menas node i have a edge to node j
    points_data: [N,M,...] N denote the number of nodes, M,... denote the attribute of each node, it's a np.ndarray data,
    elements is tf.Tensor
    edges_data: [edge_nr,K,...] adjacentmatrix each row representation a edges from this node[row](only connected edge),
     it's a tf.tensor data, elements is tf.Tensor
    axis: the axis of attribute begin, now, it must be one.
    '''
    def __init__(self,adj_mt,points_data,edges_data,edges_data_dim=None,axis=1):
        if not isinstance(adj_mt,tf.Tensor) and adj_mt is not None:
            adj_mt = tf.convert_to_tensor(adj_mt)
        if not isinstance(points_data,tf.Tensor) and points_data is not None:
            points_data = tf.convert_to_tensor(points_data)
        if not isinstance(edges_data,tf.Tensor) and edges_data is not None:
            edges_data = tf.convert_to_tensor(edges_data)
        if edges_data is None:
            assert edges_data_dim is not None,"edges_data or edges_data_dim must not be None."
        else:
            edges_data_dim = edges_data.get_shape().as_list()[-1]

        self.adj_mt = adj_mt
        if self.adj_mt.dtype is not tf.bool:
            self.adj_mt = tf.cast(self.adj_mt,tf.bool)
        self.line_mask = tf.cast(tf.reshape(self.adj_mt,[-1]),tf.bool)
        #先行后列
        self.edges_data = edges_data

        self.points_data = points_data
        p_num = wmlt.combined_static_and_dynamic_shape(points_data)[0]
        self.axis = axis
        self._points_size = p_num
        with tf.device(":/cpu:0"):
            self.point_indexs = tf.range(self._points_size)
            #[edge_nr,2]:tf.Tensor,(sender_index,receive_index] value in [0,points_nr)
            self.senders_indexs, self.receivers_indexs = self.make_edge_to_points_indexs()
            self.real_edge_nr = tf.shape(self.senders_indexs)[0]
            #self._points_size = tf.Print(self._points_size,["p_e_size",self._points_size,self.real_edge_nr])
            self.global_attr = None
            self.p2e_offset_index = self.get_offset_index_for_p2e()
            #[points_nr,2] list,value is tf.Tensor, Tensor's shape is [], tensor's value is [0,edge_nr)
            self.points_to_sedges,self.points_to_redges = self.make_points_to_edges_indexs()
        self.edges_reducer_for_points = tf.unsorted_segment_sum
        #输入为[X,edge_hiden_size],输出为[1,edge_hiden_size]
        self.edges_reducer_for_global = functools.partial(tf.reduce_mean,axis=0,keepdims=True)
        #输入为[X,node_hiden_size],输出为[1,node_hiden_size]
        self.points_reducer_for_global = functools.partial(tf.reduce_mean,axis=0,keepdims=True)

        with tf.variable_scope("default_edge_value", reuse=tf.AUTO_REUSE):
            self.default_value_s = tf.get_variable("default_edge_s",shape=[1,edges_data_dim],
                                              dtype=tf.float32,
                                              initializer=tf.zeros_initializer,trainable=True)
            self.default_value_r = tf.get_variable("default_edge_r",shape=[1,edges_data_dim],
                                              dtype=tf.float32,
                                              initializer=tf.zeros_initializer,trainable=True)
        self.use_sent_edges_for_node = True
        self.use_received_edges_for_node = True

    def get_offset_index_for_p2e(self):
        index0 = wmlt.mask_to_indices(self.line_mask)
        index1 = tf.range(self.real_edge_nr)
        offset = index0 - index1
        res = tf.scatter_nd(indices=tf.expand_dims(index0,-1),updates=offset,shape=tf.shape(self.line_mask))
        return tf.stop_gradient(res)

    def points_size(self):
        return self._points_size

    def make_edge_to_points_indexs(self):
        with tf.variable_scope("make_edge_to_points_indexs"):
            p_nr = self._points_size
            edge_nr = p_nr*p_nr
            edge_index = tf.range(edge_nr)
            edge_index = tf.boolean_mask(edge_index,self.line_mask)

            def fn(index):
                i = index // p_nr
                j = tf.mod(index, p_nr)
                return tf.concat([tf.reshape(i,[1]),tf.reshape(j,[1])],axis=0)
            
            indexs = tf.map_fn(fn,elems=(edge_index),back_prop=False)

            return tf.unstack(tf.stop_gradient(indexs),axis=1)

    def make_points_to_edges_indexs(self):
        with tf.variable_scope("make_points_to_edges_index"):
            datas = tf.ones(shape=[2,self.real_edge_nr],dtype=tf.int32)
            datas = tf.foldl(self.make_point_to_edges_indexs,elems=self.point_indexs,initializer=datas,back_prop=False)
            return tf.unstack(datas,axis=0)

    def make_point_to_edges_indexs(self,datas,i):

        s_edges_indexs = self.point_indexs+i*self.points_size()

        def to_realedge_indices(indices):
            offset = tf.gather(self.p2e_offset_index,indices)
            return indices-offset

        mask = self.adj_mt[i]
        s_edges_indexs = tf.boolean_mask(s_edges_indexs,mask)
        s_edges_indexs = to_realedge_indices(s_edges_indexs)

        r_edges_indexs = self.point_indexs*self.points_size()+i
        r_edges_indexs = tf.convert_to_tensor(r_edges_indexs)
        mask = self.adj_mt[:,i]
        r_edges_indexs = tf.boolean_mask(r_edges_indexs,mask)
        r_edges_indexs = to_realedge_indices(r_edges_indexs)
        points_to_sedges,points_to_redges = tf.unstack(datas,axis=0)
        points_to_sedges = wop.set_value(points_to_sedges,i,tf.reshape(s_edges_indexs,[-1,1]))
        points_to_redges = wop.set_value(points_to_redges,i,tf.reshape(r_edges_indexs,[-1,1]))
        return tf.stack([points_to_sedges,points_to_redges],axis=0)

    def gather_points_data_for_edges(self):
        senders = tf.gather(self.points_data,self.senders_indexs)
        receivers = tf.gather(self.points_data,self.receivers_indexs)
        return senders,receivers

    def gather_globals_data_for_edges(self):
        return tf.ones(shape=tf.concat([tf.reshape(self.real_edge_nr,[1]),[1]],axis=0),dtype=tf.float32)*self.global_attr


    def points_data_for_edge(self,i,j):
        p0 = self.points_data[i]
        p1 = self.points_data[j]
        res = tf.concat([p0,p1],axis=self.axis)
        return res

    def reduce_edges_data_for_global(self):
        res = self.edges_reducer_for_global(self.edges_data)
        return res

    def reduce_points_data_for_global(self):
        res = self.points_reducer_for_global(self.points_data)
        return res

    def update_edges(self,edge_fn,scope=None,use_global_attr=True):
        if edge_fn is None:
            return
        with tf.variable_scope(scope,default_name="UpdateEddges"):
            senders, receivers = self.gather_points_data_for_edges()
            points_data = tf.concat([senders,receivers],axis=1)
            if use_global_attr:
                globals_data = self.gather_globals_data_for_edges()
                data_in = tf.concat([points_data,self.edges_data,globals_data],axis=1)
            else:
                data_in = tf.concat([points_data, self.edges_data], axis=1)
            self.edges_data = edge_fn(data_in)

    def update_points(self,point_fn,scope=None,use_global_attr=True):
        if point_fn is None:
            return
        with tf.name_scope(scope,default_name="GatherDataForUpdatePoints"):
            datas = [self.points_data]
            if self.use_sent_edges_for_node:
                s_net = self.edges_reducer_for_points(self.edges_data,self.points_to_sedges,num_segments=self.points_size(),
                                                name="gather_sent_edges_data")
                datas.append(s_net)
            if self.use_received_edges_for_node:
                r_net = self.edges_reducer_for_points(self.edges_data, self.points_to_redges, num_segments=self.points_size(),
                                                name="gather_received_edges_data")
                datas.append(r_net)
            if use_global_attr:
                g_net = tf.tile(self.global_attr,[self.points_size(),1])
                datas.append(g_net)
            net = tf.concat(datas,axis=-1)
        self.points_data = point_fn(net)


    def update_global(self,global_fn,scope=None):
        if global_fn is None:
            return
        with tf.variable_scope(scope,default_name="UpdateGlobals"):
            edge = self.reduce_edges_data_for_global()
            point = self.reduce_points_data_for_global()
            net = tf.concat([point, edge, self.global_attr], axis=1)
            output = global_fn(net)
            self.global_attr = output

    def update(self,point_fn,edge_fn,global_fn,scopes=[None,None,None],use_global_attr=True):
        self.update_edges(edge_fn,scopes[1],use_global_attr=use_global_attr)
        self.update_points(point_fn,scopes[0],use_global_attr=use_global_attr)
        if use_global_attr:
            self.update_global(global_fn,scopes[2])
        return {"edges":self.edges_data,"nodes":self.points_data,"global":self.global_attr}

    def update_edges_independent(self,edge_fn,scope=None):
        if edge_fn is None:
            return
        with tf.variable_scope(scope,default_name="UpdateEddges"):
            self.edges_data = edge_fn(self.edges_data)

    def update_points_independent(self,point_fn,scope=None):
        if point_fn is None:
            return
        with tf.variable_scope(scope,default_name="UpdatePoints"):
            net = self.points_data
            self.points_data = point_fn(net)

    def update_global_independent(self,global_fn,scope=None):
        if global_fn is None:
            return
        with tf.variable_scope(scope,default_name="UpdateGlobals"):
            self.global_attr = global_fn(self.global_attr)

    def update_independent(self,point_fn,edge_fn,global_fn,scopes=[None,None,None]):
        self.update_edges_independent(edge_fn,scopes[1])
        self.update_points_independent(point_fn,scopes[0])
        self.update_global_independent(global_fn,scopes[2])
        return {"edges":self.edges_data,"nodes":self.points_data,"global":self.global_attr}

    def concat(self,datas,use_global_attr=True):
        if "edges" in datas and datas["edges"] is not None:
            self.edges_data = tf.concat([self.edges_data,datas["edges"]],axis=1)
        if "nodes" in datas and datas["nodes"] is not None:
            self.points_data = tf.concat([self.points_data,datas["nodes"]],axis=1)
        if use_global_attr and "global" in datas and datas["global"] is not None:
            self.global_attr = tf.concat([self.global_attr,datas["global"]],axis=1)
