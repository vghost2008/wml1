#coding=utf-8
import tensorflow as tf
from tfop import set_value
import sys
import os
from .mask_utils import iou,np_iou
import numpy as np
sys.path.append(os.path.dirname(__file__))
import semantic.visualization_utils as visu
import image_visualization as ivs
import cv2
from semantic.visualization_utils import MIN_RANDOM_STANDARD_COLORS
import basic_tftools as btf

'''
mask:[N,height,width]
labels:[N]
num_classes:() not include background 
return:
[num_classes,height,width]
'''
def sparse_mask_to_dense(mask,labels,num_classes,no_background=True):
    with tf.variable_scope("SparseMaskToDense"):
        if mask.dtype is not tf.bool:
            mask = tf.cast(mask,tf.bool)
        shape = btf.combined_static_and_dynamic_shape(mask)
        if no_background:
            out_shape = [num_classes,shape[1],shape[2]]
            labels = labels-1
        else:
            out_shape = [num_classes+1, shape[1], shape[2]]
        init_res = tf.zeros(shape=out_shape,dtype=tf.bool)

        def fn(merged_m,m,l):
            tmp_data = set_value(tensor=init_res,v=m,index=l)
            return tf.logical_or(merged_m,tmp_data)
        res = tf.foldl(lambda x,y:fn(x,y[0],y[1]),elems=(mask,labels),initializer=init_res,back_prop=False)
        return res

def batch_sparse_mask_to_dense(mask,labels,lens,num_classes,no_background=True):
    def fn(mask,labels,l):
        mask = mask[:l]
        labels = labels[:l]
        return sparse_mask_to_dense(mask,labels,num_classes,no_background=no_background)
    return tf.map_fn(lambda x:fn(x[0],x[1],x[2]),elems=[mask,labels,lens],back_prop=False,dtype=tf.bool)

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

def tf_summary_image_with_mask_and_boxes(image, mask, bboxes,color=None,alpha=0.4,no_first_mask=False,name='summary_image_with_mask'):
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
        bboxes = tf.expand_dims(bboxes,axis=0)
        image = tf.cast(image,tf.float32)/255
        image = tf.image.draw_bounding_boxes(image, bboxes)
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



'''
masks:[X,H,W]
labels:[X]
output:
[num_classes,H,W]/[num_classes-1,H,W]
'''
def merge_masks(masks,labels,num_classes,size=None,no_background=True):
    if size is not None:
        width = size[1]
        height = size[0]
    elif len(masks.shape)>=3:
        width = masks.shape[2]
        height = masks.shape[1]

    if no_background:
        get_label = lambda x:max(0,x-1)
        res = np.zeros([num_classes-1,height,width],dtype=np.int32)
    else:
        get_label = lambda x:x
        res = np.zeros([num_classes,height,width],dtype=np.int32)

    for i,mask in enumerate(masks):
        label = get_label(labels[i])
        res[label:label+1,:,:] = np.logical_or(res[label:label+1,:,:],np.expand_dims(mask,axis=0))

    return res

def get_fullsize_merged_mask(masks,bboxes,labels,size,num_classes,no_background=True):
    fullsize_masks = ivs.get_fullsize_mask(bboxes,masks,size)
    return merge_masks(fullsize_masks,labels,num_classes,size,no_background)

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
