#coding=utf-8
from multiprocessing import Pool
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

'''def edit_distance(v0,v1):
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
    return min(min(c0,c1),c2)'''


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

