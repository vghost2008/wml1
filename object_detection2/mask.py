#coding=utf-8
import numpy as np
import tensorflow as tf
import object_detection2.bboxes as bboxes
import basic_tftools as btf

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

'''
mask: [N,H,W] or [B,N,H,W]
'''
@btf.add_name_scope
def resize_mask(mask,size):
    if isinstance(size,int):
        size = (size,size)
    old_type = mask.dtype
    mask = tf.expand_dims(mask,axis=-1)
    if len(mask.shape) == 5:
        old_shape = btf.combined_static_and_dynamic_shape(mask)
        mask = tf.reshape(mask,[old_shape[0]*old_shape[1]]+old_shape[2:])
    else:
        old_shape = None
    mask = tf.image.resize_bilinear(mask,size)
    if old_shape is not None:
        mask = tf.reshape(mask,[old_shape[0],old_shape[1],size[0],size[1]])
    else:
        mask = tf.squeeze(mask,axis=-1)
    
    if mask.dtype != old_type:
        mask = tf.cast(mask+0.5,old_type)
    
    return mask
        
        
