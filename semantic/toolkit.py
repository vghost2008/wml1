#coding=utf-8
import tensorflow as tf
from wtfop.wtfop_ops import set_value
import sys
import os
from .mask_utils import iou,np_iou
import numpy as np
sys.path.append(os.path.dirname(__file__))
import semantic.visualization_utils as visu
from semantic.visualization_utils import MIN_RANDOM_STANDARD_COLORS

'''
mask:[N,height,width]
labels:[N]
num_classes:()
return:
[num_classes,height,width]
'''
'''def sparse_mask_to_dense(mask,labels,num_classes):
    with tf.variable_scope("SparseMaskToDense"):
        if mask.dtype is not tf.bool:
            mask = tf.cast(mask,tf.bool)
        if mask.get_shape().is_fully_defined():
            shape = mask.get_shape().as_list()
        else:
            shape = tf.shape(mask)

        out_shape = [num_classes,shape[0],shape[1]]
        init_res = tf.zeros(shape=out_shape,dtype=tf.bool)
        nr = tf.shape(labels)[0]
        indexs = tf.range(nr)

        mask = tf.transpose(mask,perm=[2,0,1])
        def fn(data,index):
            tmp_data = set_value(tensor=init_res,v=mask[index],index=labels[index])
            return tf.logical_or(data,tmp_data)
        res = tf.foldl(fn,elems=indexs,initializer=None,back_prop=False)
        res = tf.transpose(res,perm=[1,2,0])
        return res'''

'''
image:[height,width,3]
mask:[height,width,N]
colors:[N], string
alpha:
'''
def tf_summary_image_with_mask(image, mask, color=None,alpha=0.4,no_first_mask=False,name='summary_image_with_mask'):
    with tf.device("/cpu:0"):
        if no_first_mask:
            mask = mask[:,:,1:]

        if color is None:
            with tf.device("/cpu:0"):
                mask_nr = tf.shape(mask)[2]
                color_nr = len(MIN_RANDOM_STANDARD_COLORS)
                color_tensor = tf.convert_to_tensor(MIN_RANDOM_STANDARD_COLORS)
                color = tf.gather(color_tensor,
                                  tf.mod(tf.range(mask_nr,dtype=tf.int32), color_nr))
        image = visu.tf_draw_masks_on_image(image=image,mask=mask,color=color,alpha=alpha)
        if image.get_shape().ndims ==3:
            image = tf.expand_dims(image,axis=0)
        tf.summary.image(name, image)

'''
image:[batch_size,height,width,3]
mask:[batch_size,height,width,N]
alpha:
'''
def tf_summary_images_with_masks(image, mask,alpha=0.4,colors=MIN_RANDOM_STANDARD_COLORS,name='summary_image_with_mask',no_first_mask=False,max_outputs=3):
    with tf.device("/cpu:0"):
        color_nr = len(colors)
        color_tensor = tf.convert_to_tensor(colors)
        if no_first_mask:
            mask = mask[:,:,:,1:]

        def fn(img,msk):
            mask_nr = tf.shape(msk)[2]
            color = tf.gather(color_tensor,
                              tf.mod(tf.range(mask_nr,dtype=tf.int32),color_nr))
            return visu.tf_draw_masks_on_image(image=img,mask=msk,color=color,alpha=alpha)

        images = tf.map_fn(lambda x:fn(x[0],x[1]),elems=(image,mask),dtype=tf.uint8,back_prop=False)
        tf.summary.image(name, images,max_outputs=max_outputs)

'''
image:[height,width,3]
mask:[height,width,N]
colors:[N], string
alpha:
'''
def np_draw_masks_on_image(image, mask, colors, alpha=0.4):
    if image.dtype is not np.uint8:
        image = image.astype(np.uint8)
    if mask.dtype is not np.uint8:
        mask = mask.astype(np.uint8)
    mask = np.transpose(mask, axes=[2, 0, 1])
    colors_nr = len(colors)

    for i,msk in enumerate(mask):
        image = visu.draw_mask_on_image_array(image,msk,colors[i%colors_nr],alpha)

    return image

'''
image:[batch_size,height,width,3]
mask:[batch_size,height,width,N]
alpha:
'''
def np_draw_masks_on_images(image,mask,alpha,colors=MIN_RANDOM_STANDARD_COLORS,no_first_mask=False):
    if no_first_mask:
        mask = mask[:,:,:,1:]
    res_images = []

    for img,msk in zip(image,mask):
        new_img = np_draw_masks_on_image(image=img,mask=msk,colors=colors,alpha=alpha)
        res_images.append(new_img)

    return np.array(res_images)

class ModelPerformance:
    def __init__(self,no_first_class=True):
        self.test_nr = 0
        self.total_iou = 0.
        self.no_first_class = no_first_class


    def clear(self):
        self.test_nr = 0
        self.total_iou = 0.

    '''
    mask_gt: [batch_size,h,w,num_classes]
    mask_pred: [batch_size,h,w,num_classes]
    background is [:,:,0]
    '''
    def __call__(self, mask_gt,mask_pred):
        if self.no_first_class:
            mask_gt = mask_gt[:,:,:,1:]
            mask_pred = mask_pred[:,:,:,1:]
        tmp_iou = np_iou(mask_gt,mask_pred)
        self.total_iou += tmp_iou
        self.test_nr += 1
        return tmp_iou, self.mIOU()

    def mIOU(self):
        return self.total_iou/self.test_nr
