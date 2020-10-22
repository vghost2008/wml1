#coding=utf-8
import tensorflow as tf
from object_detection2.modeling.matcher import Matcher
from object_detection2.standard_names import *
from wmodule import WModule
from basic_tftools import batch_gather
import numpy as np
import object_detection2.bboxes as odb
import img_utils as wmli


def get_img_size_from_batched_inputs(inputs):
    image = inputs[IMAGE]
    return tf.shape(image)[1:3]

class RemoveSpecifiedResults(object):
    def __init__(self,pred_fn=None):
       self.matcher = Matcher(thresholds=[1e-4],parent=self) 
       self.pred_fn = pred_fn if pred_fn is not None else self.__pred_fn
        
        
    def __call__(self, gbboxes,glabels,bboxes,labels):
        '''
        
        :param gbboxes: [1,X,4]
        :param glabels: [1,X]
        :param bboxes: [1,Y,4]
        :param labels: [1,Y]
        :return: 
        '''
        glength = tf.reshape(tf.shape(glabels)[1],[1])
        removed_gbboxes = self.pred_fn(gbboxes,glabels)
        labels, scores, indices = self.matcher(bboxes,gbboxes,glabels,glength)
        is_bboxes_removed = tf.batch_gather(removed_gbboxes,tf.nn.relu(indices))
        is_bboxes_removed = tf.logical_or(is_bboxes_removed,tf.less(indices,0))
        gbboxes = tf.expand_dims(tf.boolean_mask(gbboxes,removed_gbboxes),0)
        glabels = tf.expand_dims(tf.boolean_mask(glabels,removed_gbboxes),0)
        bboxes = tf.expand_dims(bboxes,is_bboxes_removed)
        labels = tf.expand_dims(labels,is_bboxes_removed)
        return gbboxes,glabels,bboxes,labels,removed_gbboxes,is_bboxes_removed

    @staticmethod
    def __pred_fn(gbboxes,glabels):
        return tf.zeros_like(glabels,dtype=tf.bool)


def replace_with_gtlabels(bboxes,labels,length,gtbboxes,gtlabels,gtlength,threshold=0.5):
    parent = WModule(cfg=None)
    matcher = Matcher(
        thresholds=[threshold],
        allow_low_quality_matches=False,
        cfg=None,
        parent=parent,
    )
    n_labels,_,_ = matcher(bboxes,gtbboxes,gtlabels,gtlength)
    labels = tf.where(tf.greater(n_labels,0),n_labels,labels)
    return labels

def replace_with_gtbboxes(bboxes,labels,length,gtbboxes,gtlabels,gtlength,threshold=0.5):
    parent = WModule(cfg=None)
    matcher = Matcher(
        thresholds=[threshold],
        allow_low_quality_matches=False,
        cfg=None,
        parent=parent,
    )
    n_labels,_,indices = matcher(bboxes,gtbboxes,gtlabels,gtlength)
    n_labels = tf.expand_dims(n_labels,axis=-1)
    n_labels = tf.tile(n_labels,[1,1,4])
    n_bboxes = batch_gather(gtbboxes,tf.nn.relu(indices))
    bboxes = tf.where(tf.greater(n_labels,0),n_bboxes,bboxes)
    return bboxes

'''
image_data:[h,w,c]
bboxes:[N,4] absolute coordinate
rect:[ymin,xmin,ymax,xmax) absolute coordinate
'''
def cut_bboxes(bboxes,labels,img,rect,threshold=0.5,fill_color=None,is_sub_img=False):
    res_bboxes = []
    res_labels = []

    if not isinstance(labels,np.ndarray):
        labels = np.array(labels)

    remove_bboxes = []
    no_zero = 1e-3
    for i in range(labels.shape[0]):
        iou = odb.npbboxes_intersection_of_box0([bboxes[i]],rect)
        if iou<threshold and iou>no_zero:
            remove_bboxes.append(bboxes[i])
        elif iou>=threshold:
            res_bboxes.append(bboxes[i])
            res_labels.append(labels[i])

    if not is_sub_img:
        img = wmli.sub_image(img,rect)

    if fill_color is not None and len(remove_bboxes)>0:
        remove_bboxes = np.stack(remove_bboxes, axis=0) - np.array([[rect[0], rect[1], rect[0], rect[1]]])
        remove_bboxes = remove_bboxes.astype(np.int32)
        img = wmli.remove_boxes_of_img(img,remove_bboxes,default_value=fill_color)

    if len(res_labels)>0:
        res_bboxes = np.stack(res_bboxes,axis=0) - np.array([[rect[0],rect[1],rect[0],rect[1]]])
        res_labels = np.array(res_labels)
    else:
        res_bboxes = np.zeros(shape=[0,4],dtype=bboxes.dtype)
        res_labels = np.zeros(shape=[0],dtype=labels.dtype)

    return res_bboxes,res_labels,img

'''
在每一个标目标附近裁剪出一个子图
bboxes: [N,4] absolute coordinate
size:[H,W]
return:
[N,4] (ymin,xmin,ymax,xmax) absolute coordinate
'''
def get_random_cut_bboxes_rect(bboxes,size,img_size):
    res = []
    y_max,x_max = img_size[0],img_size[1]
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes)
    if bboxes.shape[0] == 0:
        return []
    obj_ann_bboxes = odb.expand_bbox_by_size(bboxes,[x//2 for x in size],format='yxminmax')
    obj_ann_bboxes = odb.to_xyminwh(obj_ann_bboxes)


    for t_bbox in obj_ann_bboxes:
        t_bbox = list(t_bbox)
        t_bbox[1] = max(0,min(t_bbox[1],y_max))
        t_bbox[0] = max(0,min(t_bbox[0],x_max))
        t_bbox = odb.random_bbox_in_bbox(t_bbox,size)
        rect = (t_bbox[1],t_bbox[0],t_bbox[1]+t_bbox[3],t_bbox[0]+t_bbox[2])

        res.append(rect)
    return res

'''
在每一个标目标附近裁剪出一个子图, 如果一个bbox已经出现在前面的某一个rect中则跳过
用于保证一个instance仅在结果中出现一次
bboxes: [N,4] absolute coordinate
size:[H,W]
return:
[N,4] (ymin,xmin,ymax,xmax) absolute coordinate
'''
def get_random_cut_bboxes_rectv2(bboxes,size,img_size,labels=None,force_cut_labels=None):
    res = []
    y_max,x_max = img_size[0],img_size[1]
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes)
    if bboxes.shape[0] == 0:
        return []
    obj_ann_bboxes = odb.expand_bbox_by_size(bboxes,[x//2 for x in size],format='yxminmax')
    obj_ann_bboxes = odb.to_xyminwh(obj_ann_bboxes)


    for i,t_bbox in enumerate(obj_ann_bboxes):
        bbox = bboxes[i]
        if len(res)>0:
            if labels is None or (labels[i] not in force_cut_labels):
                ious = odb.npbboxes_intersection_of_box0(bbox,res)
                max_index = np.argmax(ious)
                if ious[max_index]>0.9:
                    continue
        t_bbox = list(t_bbox)
        t_bbox[1] = max(0,min(t_bbox[1],y_max))
        t_bbox[0] = max(0,min(t_bbox[0],x_max))
        t_bbox = odb.random_bbox_in_bbox(t_bbox,size)
        rect = (t_bbox[1],t_bbox[0],t_bbox[1]+t_bbox[3],t_bbox[0]+t_bbox[2])

        res.append(rect)
    return res

def filter_by_classeswise_thresholds(labels,bboxes,probs,thresholds):
    '''

    :param labels: [N]
    :param bboxes: [N,4]
    :param probs: [N]
    :param thresholds: 不包含背景0
    :return:
    '''
    n_labels = []
    n_bboxes = []
    n_probs = []

    for i,l in enumerate(labels):
        tp = thresholds[l-1]
        p = probs[i]
        if tp<=p:
            n_labels.append(l)
            n_bboxes.append(bboxes[i])
            n_probs.append(p)

    return np.array(n_labels),np.array(n_bboxes),np.array(n_probs)
