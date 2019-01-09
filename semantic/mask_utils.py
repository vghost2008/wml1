#coding=utf-8
import tensorflow as tf
import numpy as np

def iou(mask0,mask1):
    if mask0.dtype is not tf.bool:
        mask0 = tf.cast(mask0,tf.bool)
    if mask1.dtype is not tf.uint8:
        mask1 = tf.cast(mask1,tf.bool)

    if not mask0.get_shape().is_compatible_with(mask1.get_shape()):
        print("Mask not compatible with each other")
        return tf.constant(0.,tf.float32)

    different = tf.logical_xor(mask0,mask1)
    different = tf.cast(different,tf.float32)
    different = tf.reduce_sum(different)

    union = tf.logical_or(mask0,mask1)
    union = tf.cast(union,tf.float32)
    union = tf.reduce_sum(union)

    if union == 0:
        return 100.0

    return 100.0-different*100.0/union

def np_iou(mask0,mask1):
    if mask0.dtype is not np.bool:
        mask0 = mask0.astype(np.bool)
    if mask1.dtype is not tf.bool:
        mask1 = mask1.astype(np.bool)

    if len(mask0.shape) != len(mask1.shape):
        print("Mask not compatible with each other")
        return 0.

    different = np.logical_xor(mask0,mask1)
    different = different.astype(np.float32)
    different = np.sum(different)

    union = np.logical_or(mask0,mask1)
    union = union.astype(np.float32)
    union = np.sum(union)
    print("union={}, different={}, mask={}, gt={}".format(union,different,np.sum(mask0.astype(np.float32)),np.sum(mask1.astype(np.float32))))

    if union == 0:
        return 100.0

    return 100.0-different*100.0/union

'''
mask:[H,W,NUM_CLASSES]
mask:[H,W]
'''
def np_mask2masklabels(mask,begin_label=1):
    res = np.zeros(mask.shape[:2],np.int32)
    h = mask.shape[0]
    w = mask.shape[1]
    num_classes = mask.shape[2]

    for i in range(h):
        for j in range(w):
            for k in range(num_classes):
                if mask[i,j,k]>0:
                    res[i,j] = k+begin_label
                    break

    return res



