import os
import sys
import numpy as np

def to_one_hot(value,class_num=10,max_value=1.,min_value=0.,dtype=int):
    result = np.ndarray(shape=[class_num],dtype=dtype)
    for i in range(result.shape[0]):
        result[i] = min_value
    result[value] = max_value
    return result
