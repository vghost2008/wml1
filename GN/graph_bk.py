#coding=utf-8
from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf

class Point:
    def __init__(self,index,s_edges=[],r_edges=[],axis=1):
        self.index = index
        self.s_edges = s_edges
        self.r_edges = r_edges
        self.axis = axis

    def avg_edges(self,edges_data,mode=0):
        if mode==0:
            edges = []
            for e in self.s_edges:
                edges.append(edges_data[e])
            for e in self.r_edges:
                edges.append(edges_data[e])
            edge = tf.concat(edges,axis=self.axis)
            res = tf.reduce_mean(edge,axis=self.axis,keepdims=True)
            return res
        else:
            s_edges = []
            for e in self.s_edges:
                s_edges.append(edges_data[e])
            s_edge = tf.concat(s_edges,axis=self.axis)
            s_edge = tf.reduce_mean(s_edge,axis=self.axis,keepdims=True)

            r_edges = []
            for e in self.r_edges:
                r_edges.append(edges_data[e])
            r_edge = tf.concat(r_edges,axis=self.axis)
            r_edge = tf.reduce_mean(r_edge,axis=self.axis,keepdims=True)

            res = tf.concat([s_edge,r_edge],axis=self.axis+1)
            return res


class Points:
    __metaclass__ = ABCMeta
    def __init__(self,size):
        self.size = size

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class edge:
    __metaclass__ = ABCMeta
    def __init__(self,s_index,r_index,axis=1):
        self.s_index = s_index
        self.r_index = r_index
        self.axis = axis

    def concat_points(self,points_data):
        p0 = points_data(self.s_index)
        p1 = points_data(self.r_index)
        return tf.concat([p0,p1],axis=self.axis)

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


class DynamicAdjacentMatrix:
    '''
    adj_mt: used to indict connection, row i and col j menas node i have a edge to node j
    points_data: [N,M,...] N denote the number of nodes, M,... denote the attribute of each node, it's a np.ndarray data,
    elements is tf.Tensor
    edges_data: [N*N,K,...] adjacentmatrix each row representation a edges from this node[row],
     it's a tf.tensor data, elements is tf.Tensor
    axis: the axis of attribute begin, now, it must be one.
    '''
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

        s_edge = tf.gather(self.edges_data,np.array(s_edges_indexs))
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
        r_edge = tf.gather(self.edges_data,np.array(r_edges_indexs))
        r_edge = tf.boolean_mask(r_edge,mask)
        r_edge = tf.expand_dims(r_edge,axis=0)
        r_edge = tf.reduce_mean(r_edge,axis=self.axis,keepdims=False)

        res = tf.concat([s_edge,r_edge],axis=self.axis)
        return res

    def avg_edges_data(self):
        edge = tf.boolean_mask(self.edges_data,self.line_mask)
        edge = tf.expand_dims(edge,axis=0)
        res = tf.reduce_mean(edge, axis=self.axis, keepdims=False)
        return res

    def avg_points_data(self):
        res = tf.reduce_mean(self.points_data, axis=self.axis-1, keepdims=False)
        return res