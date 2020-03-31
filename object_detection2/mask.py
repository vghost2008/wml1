#coding=utf-8
import numpy as np
import tensorflow as tf
import object_detection2.bboxes as bboxes

'''
mask: [N,h,w]仅实例的box内部部分的mask, 值为1或0
boxes: [N,4] relative coordinate
size: [2]={H,W}
'''
def mask_area_by_instance_mask(mask,boxes,size):
    shape = tf.shape(mask)
    mask = tf.reshape(mask,[shape[0],shape[1]*shape[2]])
    mask = tf.cast(mask,tf.float32)
    mask = tf.reduce_sum(mask,axis=1)
    mask = mask/tf.cast(shape[1]*shape[2],tf.float32)
    boxes_area = bboxes.box_area(boxes)*tf.cast(size[0]*size[1],tf.float32)
    return mask*boxes_area

'''
mask: [N,H,W] mask为整个图像的大小，值域为0或1
'''
def mask_area(mask):
    shape = tf.shape(mask)
    mask = tf.reshape(mask,[shape[0],shape[1]*shape[2]])
    mask = tf.cast(mask,tf.float32)
    area = tf.reduce_sum(mask,axis=1)
    return area
