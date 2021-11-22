#coding=utf-8
from multiprocessing import Pool
import tensorflow as tf
import numpy as np
import math

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

def points_to_polygon(points):
    '''

    Args:
        points: [N,2],(x,y)

    Returns:
        idxs,[N],sorted points[N,2]
    '''

    points = np.array(points)
    base_point = 0
    if points.shape[0]<=3:
        return list(range(points.shape[0])),points
    for i in range(points.shape[0]):
        if points[i,1]<points[base_point,1]:
            base_point = i
        elif points[i, 1] == points[base_point, 1] and points[i,0]<points[base_point,0]:
            base_point = i

    angles = np.zeros([points.shape[0]],dtype=np.float32)

    for i in range(points.shape[0]):
        y = points[i,1]-points[base_point,1]
        x = points[i,0]-points[base_point,0]
        angles[i] = math.atan2(y,x)
        if angles[i]<0:
            angles[i] += math.pi
    angles[base_point] = -1e-8
    idxs = np.argsort(angles)
    return idxs,points[idxs]

def left_shift_array(array,size=1):
    '''

    Args:
        array: [N]
        size: 1->N-1
    example:
        array = [1,2,3,4]
        size=1
        return:
        [2,3,4,1]
    Returns:
        [N]
    '''
    first_part = array[size:]
    second_part = array[:size]
    return np.concatenate([first_part,second_part],axis=0)

def right_shift_array(array, size=1):
    '''

    Args:
        array: [N]
        size: 1->N-1
    example:
        array = [1,2,3,4]
        size=1
        return:
        [4,1,2,3,]
    Returns:
        [N]
    '''
    first_part = array[-size:]
    second_part = array[:-size]
    return np.concatenate([first_part, second_part], axis=0)


def sign_point_line(point,line):
    '''

    Args:
        point: [2] x,y
        line: np.array([2,2]) [(x0,y0),(x1,y1)]

    Returns:
        True or False
    '''
    line = np.array(line)
    p0 = line[0]
    vec0 = line[1]-p0
    vec1 = point-p0
    return vec0[0]*vec1[1]-vec0[1]*vec1[0]<0
