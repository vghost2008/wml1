#coding=utf-8
from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
import wml_tfutils as wmlt

class AdjacentMatrix:
    '''
    points_data: [N,batch_size,M,...] N denote the number of nodes, M,... denote the attribute of each node, it's a np.ndarray data,
    elements is tf.Tensor
    edges_data: [N,N,batch_size,K,...] adjacentmatrix each row representation a edges from this node[row], if edges_data[i][j] is
    None, that means node[i] have no directly edges to node[j], it's a np.ndarray data, elements is tf.Tensor
    axis: the axis of attribute begin, now, it must be one.
    '''
    def __init__(self,points_data,edges_data,axis=1):
        self.edges_data = edges_data
        e_shape = edges_data.shape
        assert e_shape[0]==e_shape[1],"error edges data shape."


        self.points_data = points_data
        p_num = len(points_data)
        assert p_num==e_shape[1],"error Points data shape."
        self.axis = axis
        self._points_size = p_num

    def edges_size(self):
        return np.count_nonzero(self.edges_data.shape!=None)

    def points_size(self):
        return self._points_size

    def points_data_for_edge(self,i,j):
        p0 = self.points_data[i]
        p1 = self.points_data[j]
        res = tf.concat([p0,p1],axis=self.axis)
        return res

    def avg_edges_data_for_point(self,i,mode=0):
        if mode==0:
            edges = []
            for j in range(self._points_size):
                edge_data = self.edges_data[i,j]
                if edge_data is None:
                    continue
                edges.append(edge_data)

            for j in range(self._points_size):
                if j==i:
                    continue
                edge_data = self.edges_data[j,i]
                if edge_data is None:
                    continue
                edges.append(edge_data)

            if len(edges)==0:
                return None

            edge = tf.stack(edges,axis=self.axis)
            res = tf.reduce_mean(edge,axis=self.axis,keepdims=False)
            return res
        else:
            s_edges = []
            for j in range(self._points_size):
                edge_data = self.edges_data[i,j]
                if edge_data is None:
                    continue
                s_edges.append(edge_data)
            if len(s_edges)>0:
                s_edge = tf.stack(s_edges,axis=self.axis)
                s_edge = tf.reduce_mean(s_edge,axis=self.axis,keepdims=False)
            else:
                s_edge = tf.zeros_like(self.points_data[0])

            r_edges = []
            for j in range(self._points_size):
                if j==i:
                    continue
                edge_data = self.edges_data[j,i]
                if edge_data is None:
                    continue
                s_edges.append(edge_data)
            if len(r_edges)>0:
                r_edge = tf.stack(r_edges,axis=self.axis)
                r_edge = tf.reduce_mean(r_edge,axis=self.axis,keepdims=False)
            else:
                r_edge = tf.zeros_like(self.points_data[0])

            if len(s_edges)==0 and len(r_edges)==0:
                return None

            res = tf.concat([s_edge,r_edge],axis=self.axis)
            return res

    def avg_edges_data(self):
        edges = []
        for i in range(self._points_size):
            for j in range(self._points_size):
                edge_data = self.edges_data[i,j]
                if edge_data is None:
                    continue
                edges.append(edge_data)

        if len(edges) == 0:
            return None

        edge = tf.stack(edges, axis=self.axis)
        res = tf.reduce_mean(edge, axis=self.axis, keepdims=False)
        return res

    def avg_points_data(self):
        point = tf.stack(self.points_data, axis=self.axis)
        res = tf.reduce_mean(point, axis=self.axis, keepdims=False)
        return res


'''class DynamicAdjacentMatrix:
    \'''
    adj_mt: used to indict connection, row i and col j menas node i have a edge to node j
    points_data: [N,M,...] N denote the number of nodes, M,... denote the attribute of each node, it's a np.ndarray data,
    elements is tf.Tensor
    edges_data: [N*N,K,...] adjacentmatrix each row representation a edges from this node[row],
     it's a tf.tensor data, elements is tf.Tensor
    axis: the axis of attribute begin, now, it must be one.
    \'''
    def __init__(self,adj_mt,points_data,edges_data,axis=1):
        self.adj_mt = adj_mt
        if self.adj_mt.dtype is not tf.bool:
            self.adj_mt = tf.cast(self.adj_mt,tf.bool)
        self.line_mask = tf.reshape(self.adj_mt,[-1,1])
        self.edges_data = edges_data

        self.points_data = points_data
        p_num = points_data.get_shape().as_list()[0]
        self.axis = axis
        self._points_size = p_num

    def points_size(self):
        return self._points_size

    def points_data_for_edge(self,i,j):
        p0 = self.points_data[i]
        p1 = self.points_data[j]
        res = tf.concat([p0,p1],axis=self.axis)
        return res

    def avg_edges_data_for_point(self,i):
        s_edges_indexs = []
        for j in range(self._points_size):
            index = i*self.points_size()+j
            s_edges_indexs.append(index)

        s_edge = self.edges_data.gather(np.array(s_edges_indexs))
        #s_edge = tf.gather(self.edges_data,np.array(s_edges_indexs))
        mask = self.adj_mt[i]
        mask = tf.expand_dims(mask,axis=1)
        s_edge = tf.boolean_mask(s_edge,mask)
        s_edge = tf.expand_dims(s_edge,axis=0)
        s_edge = tf.reduce_mean(s_edge,axis=self.axis,keepdims=False)

        r_edges_indexs = []
        for j in range(self._points_size):
            index = j*self.points_size()+i
            r_edges_indexs.append(index)
        mask = self.adj_mt[:,i]
        mask = tf.expand_dims(mask,axis=1)
        r_edge = self.edges_data.gather(np.array(r_edges_indexs))
        #r_edge = tf.gather(self.edges_data,np.array(r_edges_indexs))
        r_edge = tf.boolean_mask(r_edge,mask)
        r_edge = tf.expand_dims(r_edge,axis=0)
        r_edge = tf.reduce_mean(r_edge,axis=self.axis,keepdims=False)

        res = tf.concat([s_edge,r_edge],axis=self.axis)
        return res

    def avg_edges_data(self):
        edges_data = self.edges_data.stack()
        #edges_data = self.edges_data
        edge = tf.boolean_mask(edges_data,self.line_mask)
        edge = tf.expand_dims(edge,axis=0)
        res = tf.reduce_mean(edge, axis=self.axis, keepdims=False)
        return res

    def avg_points_data(self):
        res = tf.reduce_mean(self.points_data, axis=self.axis-1, keepdims=False)
        return res
'''
class DynamicAdjacentMatrix:
    '''
    adj_mt: used to indict connection, row i and col j menas node i have a edge to node j
    points_data: [N,M,...] N denote the number of nodes, M,... denote the attribute of each node, it's a np.ndarray data,
    elements is tf.Tensor
    edges_data: [edge_nr,K,...] adjacentmatrix each row representation a edges from this node[row](only connected edge),
     it's a tf.tensor data, elements is tf.Tensor
    axis: the axis of attribute begin, now, it must be one.
    '''
    def __init__(self,adj_mt,points_data,edges_data,axis=1):
        if not isinstance(adj_mt,tf.Tensor) and adj_mt is not None:
            adj_mt = tf.convert_to_tensor(adj_mt)
        if not isinstance(points_data,tf.Tensor) and points_data is not None:
            points_data = tf.convert_to_tensor(points_data)
        if not isinstance(edges_data,tf.Tensor) and edges_data is not None:
            edges_data = tf.convert_to_tensor(edges_data)
        self.adj_mt = adj_mt
        if self.adj_mt.dtype is not tf.bool:
            self.adj_mt = tf.cast(self.adj_mt,tf.bool)
        self.line_mask = tf.cast(tf.reshape(self.adj_mt,[-1]),tf.bool)
        #先行后列
        self.edges_data = edges_data

        self.points_data = points_data
        p_num = points_data.get_shape().as_list()[0]
        self.axis = axis
        self._points_size = p_num
        #[edge_nr,2]:tf.Tensor,(sender_index,receive_index] value in [0,points_nr)
        self.edges_to_points_index = self.make_edge_to_points_indexs()
        self.real_edge_nr = tf.shape(self.edges_to_points_index)[0]
        self.global_attr = None
        #[points_nr,2] list,value is tf.Tensor, Tensor's shape is [], tensor's value is [0,edge_nr)
        self.points_to_edges_index = self.make_points_to_edges_indexs()

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

            return indexs

    def make_points_to_edges_indexs(self):
        with tf.variable_scope("make_points_to_edges_index"):
            res = []
            for i in range(self._points_size):
                res.append(self.make_point_to_edges_indexs(i))
            return res

    def make_point_to_edges_indexs(self,i):
        def to_realedge_indices(indices):
            mask = wmlt.indices_to_dense_vector(indices=indices,size=self._points_size*self._points_size,indices_value=1,dtype=tf.int32)
            mask = tf.boolean_mask(mask,self.line_mask)
            return wmlt.mask_to_indices(mask)

        s_edges_indexs = []
        for j in range(self._points_size):
            index = i*self.points_size()+j
            s_edges_indexs.append(index)

        #s_edge = self.edges_data.gather(np.array(s_edges_indexs))
        s_edges_indexs = tf.convert_to_tensor(s_edges_indexs)
        mask = self.adj_mt[i]
        s_edges_indexs = tf.boolean_mask(s_edges_indexs,mask)
        s_edges_indexs = to_realedge_indices(s_edges_indexs)

        r_edges_indexs = []
        for j in range(self._points_size):
            index = j*self.points_size()+i
            r_edges_indexs.append(index)
        r_edges_indexs = tf.convert_to_tensor(r_edges_indexs)
        mask = self.adj_mt[:,i]
        r_edges_indexs = tf.boolean_mask(r_edges_indexs,mask)
        r_edges_indexs = to_realedge_indices(r_edges_indexs)
        res = [s_edges_indexs,r_edges_indexs]
        return res

    def gather_points_data_for_edges(self):
        senders_indexs,receivers_indexs = tf.unstack(self.edges_to_points_index,axis=1)
        senders = tf.gather(self.points_data,senders_indexs)
        receivers = tf.gather(self.points_data,receivers_indexs)
        return senders,receivers

    def gather_globals_data_for_edges(self):
        return tf.ones(shape=tf.concat([tf.reshape(self.real_edge_nr,[1]),[1]],axis=0),dtype=tf.float32)*self.global_attr


    def points_data_for_edge(self,i,j):
        p0 = self.points_data[i]
        p1 = self.points_data[j]
        res = tf.concat([p0,p1],axis=self.axis)
        return res


    def avg_edges_data_for_point(self,i):
        s_edges_indexs,r_edges_indexs = self.points_to_edges_index[i]
        with tf.variable_scope("default_edge_value", reuse=tf.AUTO_REUSE):
            default_value_s = tf.get_variable("default_edge_s",shape=[1,self.edges_data.get_shape().as_list()[-1]],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer,trainable=True)
            default_value_r = tf.get_variable("default_edge_r",shape=[1,self.edges_data.get_shape().as_list()[-1]],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer,trainable=True)
        s_edge = DynamicAdjacentMatrix.safe_gather(self.edges_data,s_edges_indexs,default_value=default_value_s)
        r_edge = DynamicAdjacentMatrix.safe_gather(self.edges_data,r_edges_indexs,default_value=default_value_r)

        s_edge = tf.reduce_mean(s_edge,axis=0,keepdims=False)

        r_edge = tf.reduce_mean(r_edge,axis=0,keepdims=False)

        res = tf.expand_dims(tf.concat([s_edge,r_edge],axis=0),axis=0)
        return res

    def avg_edges_data(self):
        res = tf.reduce_mean(self.edges_data, axis=0, keepdims=True)
        return res

    def avg_points_data(self):
        res = tf.reduce_mean(self.points_data, axis=0, keepdims=True)
        return res

    def update_edges(self,edge_fn,scope=None):
        with tf.variable_scope(scope,default_name="UpdateEddges"):
            senders, receivers = self.gather_points_data_for_edges()
            points_data = tf.concat([senders,receivers],axis=1)
            globals_data = self.gather_globals_data_for_edges()
            data_in = tf.concat([points_data,self.edges_data,globals_data],axis=1)
            self.edges_data = edge_fn(data_in)

    def update_points(self,point_fn,scope=None):
        points_data = []
        with tf.variable_scope(scope,default_name="UpdatePoints"):
            for i in range(self._points_size):
                edges = self.avg_edges_data_for_point(i)
                point = tf.expand_dims(self.points_data[i],axis=0)
                net = tf.concat([point, edges, self.global_attr], axis=1)
                output = point_fn(net)
                points_data.append(output)
        self.points_data = tf.concat(points_data,axis=0)

    def update_global(self,global_fn,scope=None):
        with tf.variable_scope(scope,default_name="UpdateGlobals"):
            edge = self.avg_edges_data()
            point = self.avg_points_data()
            net = tf.concat([point, edge, self.global_attr], axis=1)
            output = global_fn(net)
            self.global_attr = output

    def update(self,point_fn,edge_fn,global_fn,scopes=[None,None,None]):
        self.update_edges(edge_fn,scopes[1])
        self.update_points(point_fn,scopes[0])
        self.update_global(global_fn,scopes[2])
        return {"edges":self.edges_data,"nodes":self.points_data,"global":self.global_attr}

    def update_edges_independent(self,edge_fn,scope=None):
        with tf.variable_scope(scope,default_name="UpdateEddges"):
            self.edges_data = edge_fn(self.edges_data)

    def update_points_independent(self,point_fn,scope=None):
        with tf.variable_scope(scope,default_name="UpdatePoints"):
            net = self.points_data
            self.points_data = point_fn(net)

    def update_global_independent(self,global_fn,scope=None):
        with tf.variable_scope(scope,default_name="UpdateGlobals"):
            self.global_attr = global_fn(self.global_attr)

    def update_independent(self,point_fn,edge_fn,global_fn,scopes=[None,None,None]):
        self.update_edges_independent(edge_fn,scopes[1])
        self.update_points_independent(point_fn,scopes[0])
        self.update_global_independent(global_fn,scopes[2])
        return {"edges":self.edges_data,"nodes":self.points_data,"global":self.global_attr}

    def concat(self,datas):
        if "edges" in datas and datas["edges"] is not None:
            self.edges_data = tf.concat([self.edges_data,datas["edges"]],axis=1)
        if "nodes" in datas and datas["nodes"] is not None:
            self.points_data = tf.concat([self.points_data,datas["nodes"]],axis=1)
        if "global" in datas and datas["global"] is not None:
            self.global_attr = tf.concat([self.global_attr,datas["global"]],axis=1)

    @staticmethod
    def safe_gather(params, indices,default_value):
        return tf.cond(tf.greater(tf.shape(indices)[0],0),lambda:tf.gather(params,indices),lambda:default_value)

