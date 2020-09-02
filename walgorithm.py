#coding=utf-8
from multiprocessing import Pool
import tensorflow as tf
import numpy as np

def _edit_distance(v0,v1):
    if v0 == v1:
        return 0
    if (len(v0)==0) or (len(v1)==0):
        return max(len(v0),len(v1))
    c0 = _edit_distance(v0[:-1],v1)+1
    c1 = _edit_distance(v0,v1[:-1])+1
    cr = 0
    if v0[-1] != v1[-1]:
        cr = 1
    c2 = _edit_distance(v0[:-1],v1[:-1])+cr
    return min(min(c0,c1),c2)

def mt_edit_distance(v0,v1,pool):
    if v0 == v1:
        return 0
    if (len(v0)==0) or (len(v1)==0):
        return max(len(v0),len(v1))
    c0 = edit_distance(v0[:-1],v1)+1
    c1 = edit_distance(v0,v1[:-1])+1
    cr = 0
    if v0[-1] != v1[-1]:
        cr = 1
    c2 = edit_distance(v0[:-1],v1[:-1])+cr
    return min(min(c0,c1),c2)

def edit_distance(sm, sn):
    m, n = len(sm) + 1, len(sn) + 1

    matrix = np.ndarray(shape=[m,n],dtype=np.int32)

    matrix[0][0] = 0
    for i in range(1, m):
        matrix[i][0] = matrix[i - 1][0] + 1

    for j in range(1, n):
        matrix[0][j] = matrix[0][j - 1] + 1

    for i in range(1, m):
        for j in range(1, n):
            if sm[i - 1] == sn[j - 1]:
                cost = 0
            else:
                cost = 1

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost)

    return matrix[m - 1][n - 1]


def pearsonr(x,y):
    #Pearson_correlation coefficient [-1,1]
    if not isinstance(x,np.ndarray):
        x = np.array(x)

    if not isinstance(y, np.ndarray):
        y = np.array(y)

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_ba = x-x_mean
    y_ba = y-y_mean
    v = np.sum(x_ba*y_ba)
    dx = np.sum((x-x_mean)**2)
    dy = np.sum((y-y_mean)**2)
    sv = np.sqrt(dx*dy)+1e-8

    return v/sv

def tfpearsonr(x,y):
    #Pearson_correlation coefficient [-1,1]
    x = tf.convert_to_tensor(x,dtype=tf.float32)
    y = tf.convert_to_tensor(y,dtype=tf.float32)

    x_mean = tf.reduce_mean(x,keepdims=False)
    y_mean = tf.reduce_mean(y,keepdims=False)
    x_ba = x-x_mean
    y_ba = y-y_mean
    v = tf.reduce_sum(x_ba*y_ba)
    dx = tf.reduce_sum((x-x_mean)**2)
    dy = tf.reduce_sum((y-y_mean)**2)
    sv = tf.sqrt(dx*dy)+1e-8

    return v/sv
