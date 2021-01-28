#coding=utf-8
import tensorflow as tf
import wtfop.wtfop_ops as wtfop
import basic_tftools as btf
import numpy as np
import os
import sys
import random
import basic_tftools as btf
sys.path.append(os.path.dirname(__file__))
import math
import object_detection.wmath as wmath
import wtfop.wtfop_ops as wtfop
import img_utils as wmli
import cv2 as cv
import copy

def random_distored_boxes(bboxes,limits=[0.,0.,0.],size=1,keep_org=True):
    if bboxes.get_shape().ndims == 2:
        return wtfop.random_distored_boxes(boxes=bboxes,limits=limits,size=size,keep_org=keep_org)
    else:
        return tf.map_fn(lambda x: wtfop.random_distored_boxes(boxes=x,limits=limits,size=size,keep_org=keep_org),
                         elems=bboxes,back_prop=False)

'''
bboxes:[batch_size,X,4] or [X,4]
return:
[batch_size,X]
'''
def box_area(bboxes):
    with tf.name_scope("box_area"):
        ymin,xmin,ymax,xmax = tf.unstack(bboxes,axis=len(bboxes.get_shape())-1)
        h = ymax-ymin
        w = xmax-xmin
        return tf.multiply(h,w)

def batch_nms_wrapper(bboxes,classes,lens,confidence=None,nms=None,k=200,sort=False):
    with tf.name_scope("batch_nms_wrapper"):
        if confidence is None:
            confidence = tf.ones_like(classes,dtype=tf.float32)
        def do_nms(sboxes,sclasses,sconfidence,lens):
            if sort:
                r_index = tf.range(lens)
                sconfidence,t_index = tf.nn.top_k(sconfidence[:lens],k=lens)
                sboxes = tf.gather(sboxes,t_index)
                sclasses = tf.gather(sclasses,t_index)
                r_index = tf.gather(r_index,t_index)
            boxes,labels,nms_indexs = nms(bboxes=sboxes[:lens],classes=sclasses[:lens],confidence=sconfidence[:lens])
            r_len = tf.shape(nms_indexs)[0]
            boxes = tf.pad(boxes,[[0,k-r_len],[0,0]])
            labels = tf.pad(labels,[[0,k-r_len]])
            if sort:
                nms_indexs = tf.gather(r_index,nms_indexs)
            nms_indexs = tf.pad(nms_indexs,[[0,k-r_len]])
            return [boxes,labels,nms_indexs,r_len]
        boxes,labels,nms_indexs,lens = btf.static_or_dynamic_map_fn(lambda x:do_nms(x[0],x[1],x[2],x[3]),elems=[bboxes,classes,confidence,lens],
                                                                     dtype=(tf.float32,tf.int32,tf.int32,tf.int32))
        return boxes,labels,nms_indexs,lens

'''
bbox_ref:通过一个bbox定义出在原图中的一个子图
bboxs:为在原图中的bboxs, 转换为在子图中的bboxs
'''
def bboxes_resize(bbox_ref, bboxes, name=None):
    with tf.name_scope(name, 'bboxes_resize'):
        '''
        bbox_ref 的shape为[4] [y_min,x_min,y_max,x_max]
        '''
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v
        #Scale.
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes

'''
find boxes with center point in margins
margins:[ymin,xmin,ymax,xmax]
bboxes:[N,4],ymin,xmin,ymax,xmax
'''
def bboxes_filter_center(labels, bboxes, margins=[0., 0., 0., 0.],
                         scope=None):

    with tf.name_scope(scope, 'bboxes_filter', [labels, bboxes]):
        cy = (bboxes[:, 0] + bboxes[:, 2]) / 2.
        cx = (bboxes[:, 1] + bboxes[:, 3]) / 2.
        mask = tf.greater(cy, margins[0])
        mask = tf.logical_and(mask, tf.greater(cx, margins[1]))
        mask = tf.logical_and(mask, tf.less(cx, 1. + margins[2]))
        mask = tf.logical_and(mask, tf.less(cx, 1. + margins[3]))
        labels = tf.boolean_mask(labels, mask)
        bboxes = tf.boolean_mask(bboxes, mask)
        return labels, bboxes

'''
用于处理图像被裁剪后的边缘情况
如果assign_negative=True,如果原来的box与当前图像重叠区域小于threshold的labels变为相应的负值, 否则相应的labels被删除
目前默认使用的assign_negative=False
'''
def bboxes_filter_overlap(labels, bboxes,
                          threshold=0.5, assign_negative=False,
                          scope=None,
                          return_ignore_mask=False):

    with tf.name_scope(scope, 'bboxes_filter', [labels, bboxes]):
        scores = bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype),
                                     bboxes)
        mask = scores > threshold
        if assign_negative:
            labels = tf.where(mask, labels, -labels)
        else:
            '''
            返回仅mask为True的labels,
            '''
            labels = tf.boolean_mask(labels, mask)
            bboxes = tf.boolean_mask(bboxes, mask)

        bboxes = tf_correct_yxminmax_boxes(bboxes)

        if return_ignore_mask:
            ignore_mask = tf.logical_not(mask)
            ignore_mask = tf.logical_and(ignore_mask,tf.greater_equal(scores,1e-3))
            return labels, bboxes,mask,ignore_mask
        else:
            return labels, bboxes,mask

def bboxes_filter_labels(labels, bboxes,
                         out_labels=[], num_classes=np.inf,
                         scope=None):

    with tf.name_scope(scope, 'bboxes_filter_labels', [labels, bboxes]):
        mask = tf.greater_equal(labels, num_classes)
        for l in labels:
            mask = tf.logical_and(mask, tf.not_equal(labels, l))
        labels = tf.boolean_mask(labels, mask)
        bboxes = tf.boolean_mask(bboxes, mask)
        return labels, bboxes


def bboxes_jaccard(bbox_ref, bboxes, name=None):

    with tf.name_scope(name, 'bboxes_jaccard'):
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        union_vol = -inter_vol \
            + (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1]) \
            + (bbox_ref[2] - bbox_ref[0]) * (bbox_ref[3] - bbox_ref[1])
        jaccard = wmath.safe_divide(inter_vol, union_vol, 'jaccard')
        return jaccard

'''
bbox_ref:[1,4], [[ymin,xmin,ymax,xmax]]
bboxes:[N,4],[[ymin,xmin,ymax,xmax],...]
'''
def npbboxes_jaccard(bbox_ref, bboxes, name=None):

    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    int_ymin = np.maximum(bboxes[0], bbox_ref[0])
    int_xmin = np.maximum(bboxes[1], bbox_ref[1])
    int_ymax = np.minimum(bboxes[2], bbox_ref[2])
    int_xmax = np.minimum(bboxes[3], bbox_ref[3])
    h = np.maximum(int_ymax - int_ymin, 0.)
    w = np.maximum(int_xmax - int_xmin, 0.)
    inter_vol = h * w
    union_vol = -inter_vol \
                + (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1]) \
                + (bbox_ref[2] - bbox_ref[0]) * (bbox_ref[3] - bbox_ref[1])
    jaccard = wmath.npsafe_divide(inter_vol, union_vol, 'jaccard')
    return jaccard

'''
box0:[N,4], or [1,4],[ymin,xmin,ymax,xmax],...
box1:[N,4], or [1,4]
return:
[N],返回box0,box1交叉面积占box0的百分比
'''
def npbboxes_intersection_of_box0(box0,box1):

    bbox_ref= np.transpose(box0)
    bboxes = np.transpose(box1)
    int_ymin = np.maximum(bboxes[0], bbox_ref[0])
    int_xmin = np.maximum(bboxes[1], bbox_ref[1])
    int_ymax = np.minimum(bboxes[2], bbox_ref[2])
    int_xmax = np.minimum(bboxes[3], bbox_ref[3])
    h = np.maximum(int_ymax - int_ymin, 0.)
    w = np.maximum(int_xmax - int_xmin, 0.)
    inter_vol = h * w
    union_vol = (bbox_ref[2] - bbox_ref[0]) * (bbox_ref[3] - bbox_ref[1])
    jaccard = wmath.npsafe_divide(inter_vol, union_vol, 'jaccard')
    return jaccard
'''
bboxes0:[nr,4]
bboxes1:[nr,4]
return:
[nr]
'''
def batch_bboxes_jaccard(bboxes0, bboxes1, name=None):

    with tf.name_scope(name, 'bboxes_jaccard'):
        bboxes0 = tf.transpose(bboxes0)
        bboxes1 = tf.transpose(bboxes1)
        int_ymin = tf.maximum(bboxes0[0], bboxes1[0])
        int_xmin = tf.maximum(bboxes0[1], bboxes1[1])
        int_ymax = tf.minimum(bboxes0[2], bboxes1[2])
        int_xmax = tf.minimum(bboxes0[3], bboxes1[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        union_vol = -inter_vol \
                    + (bboxes0[2] - bboxes0[0]) * (bboxes0[3] - bboxes0[1]) \
                    + (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
        jaccard = wmath.safe_divide(inter_vol, union_vol, 'jaccard')
        return jaccard

'''
返回交叉面积的百分比，面积是相对于bboxes
'''
def bboxes_intersection(bbox_ref, bboxes, name=None):
    with tf.name_scope(name, 'bboxes_intersection'):
        '''
        bboxes的shape为[N,4],转换后变为[4,N]
        bbox_ref可以为[N,4]或[4]
        '''
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
        scores = wmath.safe_divide(inter_vol, bboxes_vol, 'intersection')
        return scores
'''
将以ymin,xmin,ymax,xmax表示的box转换为以cy,cx,size,ratio表示的box
'''
def to_cxysa(data):
    data = np.reshape(data,[-1,4])
    new_shape = data.shape
    res_data = np.zeros_like(data)
    for i in range(new_shape[0]):
        cy = (data[i][0]+data[i][2])*0.5
        cx= (data[i][1] + data[i][3]) * 0.5
        width = (data[i][3]-data[i][1])
        height = (data[i][2]-data[i][0])
        size = math.sqrt(width*height)
        if width>0.0:
            ratio = height/width
        else:
            ratio = 0
        res_data[i][0] = cy
        res_data[i][1] = cx
        res_data[i][2] = size
        res_data[i][3] = ratio

    return res_data

'''
data:[X,4] (ymin,xmin,ymax,xmax)
return:
[X,4] (cy,cx,h,w)
'''
def npto_cyxhw(data):
    data = np.transpose(data)
    ymin,xmin,ymax,xmax = data[0],data[1],data[2],data[3]
    cy = (ymin+ymax)/2
    cx = (xmin+xmax)/2
    h = (ymax-ymin)/2
    w = (xmax-xmin)/2
    data = np.stack([cy,cx,h,w],axis=1)
    return data

def npminxywh_toyxminmax(data):
    if not isinstance(data,np.ndarray):
        data = np.array(data)
    x = data[...,0]
    y = data[...,1]
    w = data[...,2]
    h = data[...,3]
    xmax = x+w
    ymax = y+h
    return np.stack([y,x,ymax,xmax],axis=-1)
'''
将以ymin,xmin,ymax,xmax表示的box转换为以cy,cx,h,w表示的box
对data的shape没有限制
'''
def to_cxyhw(data):
    ymin,xmin,ymax,xmax = tf.unstack(data,axis=-1)
    cy = (ymin+ymax)/2.
    cx = (xmin+xmax)/2.
    h = ymax-ymin
    w = xmax-xmin
    data = tf.stack([cy,cx,h,w],axis=-1)
    return data
'''
将以cy,cx,h,w表示的box转换为以ymin,xmin,ymax,xmax表示的box
'''
def to_yxminmax(data):
    cy, cx, h, w = tf.unstack(data,axis=-1)
    ymin = cy-h/2.
    ymax = cy+h/2.
    xmin = cx-w/2.
    xmax = cx+w/2.
    data = tf.stack([ymin,xmin,ymax,xmax],axis=-1)

    return data

'''
input:[4]/[N,4] [ymin,xmin,ymax,xmax]
output:[xmin,ymin,width,height]
'''
def to_xyminwh(bbox,is_absolute_coordinate=True):
    if not isinstance(bbox,np.ndarray):
        bbox = np.array(bbox)
    if len(bbox.shape)>= 2:
        ymin = bbox[...,0]
        xmin = bbox[...,1]
        ymax = bbox[...,2]
        xmax = bbox[...,3]
        w = xmax-xmin
        h = ymax-ymin
        return np.stack([xmin,ymin,w,h],axis=-1)

    else:
        if is_absolute_coordinate:
            return (bbox[1],bbox[0],bbox[3]-bbox[1]+1,bbox[2]-bbox[0]+1)
        else:
            return (bbox[1],bbox[0],bbox[3]-bbox[1],bbox[2]-bbox[0])
'''
将以ymin,xmin,ymax,xmax表示的box转换为以cy,cx表示的box
对data的shape没有限制
'''
def get_bboxes_center_point(data):
    old_shape = btf.combined_static_and_dynamic_shape(data)
    old_shape[-1] = 2
    data = tf.reshape(data,[-1,4])
    ymin,xmin,ymax,xmax = tf.unstack(data,axis=1)
    cy = (ymin+ymax)/2.
    cx = (xmin+xmax)/2.
    data = tf.concat([tf.expand_dims(cy,-1),tf.expand_dims(cx,axis=-1)],axis=1)
    data = tf.reshape(data,old_shape)
    return data

'''
boxes:[...,4] ymin,xmin,ymax,xmax
scale:[hscale,wscale]
'''
@btf.add_name_scope
def scale_bboxes(bboxes,scale,correct=False):
    old_shape = btf.combined_static_and_dynamic_shape(bboxes)
    data = tf.reshape(bboxes,[-1,4])
    ymin,xmin,ymax,xmax = tf.unstack(data,axis=1)
    cy = (ymin+ymax)/2.
    cx = (xmin+xmax)/2.
    h = ymax-ymin
    w = xmax-xmin
    h = scale[0]*h
    w = scale[1]*w
    ymin = cy - h / 2.
    ymax = cy + h / 2.
    xmin = cx - w / 2.
    xmax = cx + w / 2.
    data = tf.stack([ymin, xmin, ymax, xmax], axis=1)
    data = tf.reshape(data, old_shape)
    if correct:
        data = tf_correct_yxminmax_boxes(data)
    return data

'''
boxes:[...,4] ymin,xmin,ymax,xmax
scale:float
'''
@btf.add_name_scope
def get_bboxes_patchs(bboxes,scale=0.816,correct=False):
    ymin,xmin,ymax,xmax = tf.unstack(bboxes,axis=-1)
    cy = (ymin+ymax)/2.
    cx = (xmin+xmax)/2.
    h = ymax-ymin
    w = xmax-xmin
    h = scale*h
    w = scale*w
    ymin0 = cy - h / 2.
    ymax0 = cy + h / 2.
    xmin0 = cx - w / 2.
    xmax0 = cx + w / 2.

    ymin1 = ymin
    ymax1 = ymin1+h
    xmin1 = xmin
    xmax1 = xmin+w

    ymin2 = ymin
    ymax2 = ymin1+h
    xmin2 = xmax-w
    xmax2 = xmax

    ymin3 = ymax-h
    ymax3 = ymax
    xmin3 = xmax-w
    xmax3 = xmax

    ymin4 = ymax-h
    ymax4 = ymax
    xmin4 = xmin
    xmax4 = xmin+w

    data0 = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    data1 = tf.stack([ymin0, xmin0, ymax0, xmax0], axis=-1)
    data2 = tf.stack([ymin1, xmin1, ymax1, xmax1], axis=-1)
    data3 = tf.stack([ymin2, xmin2, ymax2, xmax2], axis=-1)
    data4 = tf.stack([ymin3, xmin3, ymax3, xmax3], axis=-1)
    data5 = tf.stack([ymin4, xmin4, ymax4, xmax4], axis=-1)
    data = tf.concat([data0, data1,data2,data3,data4,data5],axis=-2)
    if correct:
        data = tf_correct_yxminmax_boxes(data)
    return data
'''
data:[N,4]
校正ymin,xmin,ymax,xmax表示的box中的不合法数据
'''
def correct_yxminmax_boxes(data,keep_size=False):
    if not keep_size:
        data = np.minimum(data,1.)
        data = np.maximum(data,0.)
    else:
        nr = data.shape[0]
        for i in range(nr):
            if data[i][0]<0.:
                data[i][2] -= data[i][0]
                data[i][0] = 0.
            if data[i][1]<0.:
                data[i][3] -= data[i][1]
                data[i][1] = 0.

            if data[i][2] > 1.:
                data[i][0] -= (data[i][2]-1.)
                data[i][2] = 1.
            if data[i][3] > 1.:
                data[i][1] -= (data[i][3]-1.)
                data[i][3] = 1.

    return data
'''
校正使用相对坐标表示的box
'''
def tf_correct_yxminmax_boxes(data):
    data = tf.clip_by_value(data,clip_value_min=0.0,clip_value_max=1.0)
    return data

def distored_boxes(bboxes,scale=[],xoffset=[],yoffset=[],keep_org=True):
    if bboxes.get_shape().ndims == 2:
        return wtfop.distored_boxes(boxes=bboxes,scale=scale,xoffset=xoffset,yoffset=yoffset,keep_org=keep_org)
    else:
        return tf.map_fn(lambda x: wtfop.distored_boxes(boxes=x,scale=scale,xoffset=xoffset,yoffset=yoffset,keep_org=keep_org),
                         elems=bboxes,back_prop=False)

'''
box：[N,4] 使用相对坐标表
sub_box:[4]/[N,4] bboxes参考区域
函数返回box在[0,0,1,1]区域的表示值
'''
def restore_sub_area(bboxes,sub_box):
    h = sub_box[...,2]-sub_box[...,0]
    w = sub_box[...,3]-sub_box[...,1]
    ymin,xmin,ymax,xmax = tf.unstack(bboxes,axis=-1)
    ymin = ymin*h+sub_box[...,0]
    ymax = ymax*h+sub_box[...,0]
    xmin = xmin*w+sub_box[...,1]
    xmax = xmax*w+sub_box[...,1]
    bboxes = tf.stack([ymin,xmin,ymax,xmax],axis=-1)
    return bboxes

'''
获取bboxes在box中的相对坐标
如:box=[5,5,10,10,]
bboxes=[[7,7,8,9]]
return:
[[2,2,3,4]]
'''
def get_boxes_relative_to_box(box,bboxes,remove_zero_size_box=False):
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes)
    if bboxes.shape[0] == 0:
        return bboxes
    ymin,xmin,ymax,xmax = np.transpose(bboxes,[1,0])
    ymin = ymin-box[0]
    xmin = xmin-box[1]
    ymax = ymax-box[0]
    xmax = xmax-box[1]
    if remove_zero_size_box:
        mask = np.logical_and((ymax-ymin)>0,(xmax-xmin)>0)
        ymin = ymin[mask]
        xmin = xmin[mask]
        ymax = ymax[mask]
        xmax = xmax[mask]
    bboxes = np.stack([ymin,xmin,ymax,xmax],axis=1)
    return bboxes


'''
boxes:[N,4],ymin,xmin,ymax,xmax
regs:[N,4]
'''
def decode_boxes(boxes,
                    regs,
                    prio_scaling=[0.1, 0.1, 0.2, 0.2]):
    assert btf.channel(boxes) == btf.channel(regs),"box channel must be 4."

    l_shape = btf.combined_static_and_dynamic_shape(boxes)
    r_shape = btf.combined_static_and_dynamic_shape(regs)

    ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
    cy = (ymin + ymax) / 2.
    cx = (xmin + xmax) / 2.
    h = ymax - ymin
    w = xmax - xmin

    if regs.get_shape().ndims == 1:
        cy = regs[0] * h * prio_scaling[0] + cy
        cx = regs[1] * w * prio_scaling[1] + cx
        h = h * tf.exp(regs[2] * prio_scaling[2])
        w = w * tf.exp(regs[3] * prio_scaling[3])
    else:
        regs = tf.reshape(regs,
                          (-1, r_shape[-1]))
        regs0, regs1, regs2, regs3 = tf.unstack(regs, axis=1)
        cy = regs0 * h * prio_scaling[0] + cy
        cx = regs1 * w * prio_scaling[1] + cx
        h = h * tf.exp(regs2 * prio_scaling[2])
        w = w * tf.exp(regs3 * prio_scaling[3])

    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin,xmin,ymax,xmax],axis=0)
    bboxes = tf.transpose(bboxes,perm=[1,0])
    bboxes = tf.reshape(bboxes, l_shape)
    bboxes = tf.clip_by_value(bboxes,0.0,1.0)
    return bboxes

'''
anchor_bboxes:[batch_size,N,4] (ymin,xmin,ymax,xmax)
regs: [batch_size,N,4] (cy,cx,h,w) regs
'''
def batch_decode_boxes(anchor_bboxes,regs,prio_scaling=[0.1,0.1,0.2,0.2]):
    batch_size,data_nr = regs.get_shape().as_list()[0:2]
    if batch_size is not None and data_nr is not None:
        old_shape = [batch_size,data_nr,4]
    else:
        old_shape = tf.shape(regs)
    if anchor_bboxes.get_shape().as_list()[0] == 1:
        anchor_bboxes = tf.squeeze(anchor_bboxes,axis=0)
        return btf.static_or_dynamic_map_fn(lambda reg:wtfop.decode_boxes1(anchor_bboxes,reg,prio_scaling=prio_scaling),
                                             elems=regs,
                                             dtype=tf.float32,back_prop=False)
    else:
        anchor_bboxes = tf.reshape(anchor_bboxes,[-1,4])
        regs = tf.reshape(regs,[-1,4])
        res = wtfop.decode_boxes1(anchor_bboxes,regs,prio_scaling=prio_scaling)
        return tf.reshape(res,old_shape)

'''
cnt:[[x,y],[x,y],...]
return the bbox of a contour
'''
def bbox_of_contour(cnt):
    all_points = np.array(cnt)
    points = np.transpose(all_points)
    x,y = np.vsplit(points,2)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    return (ymin,xmin,ymax,xmax)

'''
cnt is a contour in a image, cut the area of rect
cnt:[[x,y],[x,y],...]
rect:[ymin,xmin,ymax,xmax]
output:
the new contours in sub image
'''
def cut_contour(cnt,rect):
    bbox = bbox_of_contour(cnt)
    width = max(bbox[3],rect[3])
    height = max(bbox[2],rect[2])
    img = np.zeros(shape=(height,width),dtype=np.uint8)
    segmentation = cv.drawContours(img,[cnt],-1,color=(1),thickness=cv.FILLED)
    cuted_img = wmli.sub_image(segmentation,rect)
    contours,hierarchy = cv.findContours(cuted_img,cv.CV_RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    return contours

'''
find the contours in rect of segmentation
segmentation:[H,W]
rect:[ymin,xmin,ymax,xmax]
output:
the new contours in sub image and correspond bbox
'''
def cut_contourv2(segmentation,rect):
    org_contours,org_hierarchy = cv.findContours(segmentation,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    max_area = 1e-8
    for cnt in org_contours:
        area = cv.contourArea(cnt)
        max_area = max(max_area,area)
    cuted_img = wmli.sub_image(segmentation,rect)
    contours,hierarchy = cv.findContours(cuted_img,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    boxes = []
    ratio = []
    for cnt in contours:
        boxes.append(bbox_of_contour(cnt))
        ratio.append(cv.contourArea(cnt)/max_area)
    return contours,boxes,ratio

'''
bbox:(xmin,ymin,width,height)
'''
def random_int_in_bbox(bbox):
    x = random.randint(int(bbox[0]),int(bbox[0]+bbox[2]-1))
    y = random.randint(int(bbox[1]),int(bbox[1]+bbox[3]-1))
    return x,y

'''
bbox:(xmin,ymin,width,height)
size:(width,height) the size of return bbox
random return a box with center point in the input bbox
output:
[xmin,ymin,width,height]
'''
def random_bbox_in_bbox(bbox,size):
    x,y = random_int_in_bbox(bbox)
    xmin,ymin = max(0,x-size[0]//2),max(0,y-size[1]//2)
    return [xmin,ymin,size[0],size[1]]

'''
weights [2,x],[0] values,[1]:labels
bboxes:[N,4],[xmin,ymin,width,height]
'''
def random_bbox_in_bboxes(bboxes,size,weights=None,labels=None):
    if len(bboxes) == 0:
        return (0,0,size[0],size[1])
    if weights is not None:
        old_v = 0.0
        values = []

        for v in weights[0]:
            old_v += v
            values.append(old_v)
        random_v = random.uniform(0.,old_v)
        index = 0
        for i,v in enumerate(values):
            if random_v<v:
                index = i
                break
        label = weights[1][index]
        _bboxes = []
        for l,bbox in zip(labels,bboxes):
            if l==label:
                _bboxes.append(bbox)

        if len(_bboxes) == 0:
            return random_bbox_in_bboxes(bboxes,size)
        else:
            return random_bbox_in_bboxes(_bboxes,size)
    else:
        index = random.randint(0,len(bboxes)-1)
        return random_bbox_in_bbox(bboxes[index],size)

'''
bbox:[(xmin,ymin,width,height),....] (format="xyminwh") or [(ymin,xmin,ymax,xmax),...] (format="yxminmax")
return a list of new bbox with the size scale times of the input
'''
def expand_bbox(bboxes,scale=2,format="xyminwh"):
    if format == "xyminwh":
        res_bboxes = []
        for bbox in bboxes:
            cx,cy = bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2
            new_width = bbox[2]*scale
            new_height = bbox[3]*scale
            min_x = cx-new_width//2
            min_y = cy-new_height//2
            res_bboxes.append((min_x,min_y,new_width,new_height))

        return res_bboxes
    elif format == "yxminmax":
        if not isinstance(bboxes,np.ndarray):
            bboxes = np.array(bboxes)
        ymin = bboxes[...,0]
        xmin = bboxes[...,1]
        ymax = bboxes[...,2]
        xmax = bboxes[...,3]
        h = ymax-ymin
        cy = (ymax+ymin)/2
        w = xmax-xmin
        cx = (xmax+xmin)/2
        nh = h*scale/2
        nw = w*scale/2
        nymin = cy-nh
        nymax = cy+nh
        nxmin = cx-nw
        nxmax = cx+nw

        return np.stack([nymin,nxmin,nymax,nxmax],axis=-1)

'''
bbox:[(xmin,ymin,width,height),....] (format="xyminwh") or [(ymin,xmin,ymax,xmax),...] (format="yxminmax")
size:[H,W]
return a list of new bbox with the size 'size' 
'''
def expand_bbox_by_size(bboxes,size,format="xyminwh"):
    res_bboxes = []
    if format == "xyminwh":
        for bbox in bboxes:
            cx,cy = bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2
            new_width = size[1]
            new_height = size[0]
            min_x = max(cx-new_width//2,0)
            min_y = max(cy-new_height//2,0)
            res_bboxes.append((min_x,min_y,new_width,new_height))

        return res_bboxes
    elif format == "yxminmax":
        if not isinstance(bboxes,np.ndarray):
            bboxes = np.array(bboxes)
        ymin = bboxes[...,0]
        xmin = bboxes[...,1]
        ymax = bboxes[...,2]
        xmax = bboxes[...,3]
        cy = (ymax + ymin) / 2
        cx = (xmax + xmin) / 2
        nh = size[0]//2
        nw = size[1]//2
        nymin = cy-nh
        nymax = cy+nh
        nxmin = cx-nw
        nxmax = cx+nw

        return np.stack([nymin,nxmin,nymax,nxmax],axis=-1)
'''
bboxes:[N,4]
'''
def shrink_box(bboxes,shrink_value=[0,0,0,0]):
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes)
    if not isinstance(shrink_value,list):
        shrink_value = [shrink_value]*4
    ymin,xmin,ymax,xmax = np.transpose(bboxes)
    ymin = ymin+shrink_value[0]
    xmin = xmin+shrink_value[1]
    ymax = ymax-shrink_value[2]
    xmax = xmax-shrink_value[3]
    return np.stack([ymin,xmin,ymax,xmax],axis=1)

'''
boxes:[N,4],[ymin,xmin,ymax,xmax]
return:[ymin,xmin,ymax,xmax]
'''
def bbox_of_boxes(boxes):
    if not isinstance(boxes,np.ndarray):
        boxes = np.array(boxes)
    boxes = np.transpose(boxes)
    ymin = np.min(boxes[0])
    xmin = np.min(boxes[1])
    ymax = np.max(boxes[2])
    xmax = np.max(boxes[3])
    return [ymin,xmin,ymax,xmax]

'''
boxes:[N,4],[ymin,xmin,ymax,xmax]
return:[ymin,xmin,ymax,xmax]
'''
def tfbbox_of_boxes(boxes):
    if not isinstance(boxes,tf.Tensor):
        boxes = tf.convert_to_tensor(boxes)
    boxes = tf.transpose(boxes)
    ymin = tf.reduce_min(boxes[0])
    xmin = tf.reduce_min(boxes[1])
    ymax = tf.reduce_max(boxes[2])
    xmax = tf.reduce_max(boxes[3])
    return tf.convert_to_tensor([ymin,xmin,ymax,xmax])

'''
boxes:[N,4],[ymin,xmin,ymax,xmax]
'''
def absolutely_boxes_to_relative_boxes(boxes,width,height):
    boxes = np.transpose(boxes)
    ymin = boxes[0]/height
    xmin = boxes[1]/width
    ymax = boxes[2]/height
    xmax = boxes[3]/width
    
    return np.stack([ymin,xmin,ymax,xmax],axis=1)

'''
boxes:[N,4],[ymin,xmin,ymax,xmax]
'''
def relative_boxes_to_absolutely_boxes(boxes,width,height):
    boxes = np.transpose(boxes)
    ymin = boxes[0]*height
    xmin = boxes[1]*width
    ymax = boxes[2]*height
    xmax = boxes[3]*width

    return np.stack([ymin,xmin,ymax,xmax],axis=1)
'''
boxes:[N,4],[ymin,xmin,ymax,xmax]
'''
def relative_boxes_to_absolutely_boxesi(boxes,width,height):
    return relative_boxes_to_absolutely_boxes(boxes=boxes,width=width,height=height).astype(np.int32)

'''
boxes:[N,4],[ymin,xmin,ymax,xmax]
'''
def tfrelative_boxes_to_absolutely_boxes(boxes,width,height):
    with tf.name_scope("relative_boxes_to_absolutely_boxes"):
        if not isinstance(width,tf.Tensor):
            width = tf.convert_to_tensor(width,dtype=boxes.dtype)
        elif width.dtype != boxes.dtype:
            width = tf.cast(width,boxes.dtype)
        if not isinstance(height,tf.Tensor):
            height = tf.convert_to_tensor(height,dtype=boxes.dtype)
        elif height.dtype != boxes.dtype:
            height = tf.cast(height,boxes.dtype)
        ymin = boxes[...,0]*(height-1)
        xmin = boxes[...,1]*(width-1)
        ymax = boxes[...,2]*(height-1)
        xmax = boxes[...,3]*(width-1)

        return tf.stack([ymin,xmin,ymax,xmax],axis=-1)


'''
boxes:[N,4],[ymin,xmin,ymax,xmax]
'''
def tfabsolutely_boxes_to_relative_boxes(boxes, width, height):
    with tf.name_scope("absolutely_boxes_to_relative_boxes"):
        #[B,N,4]
        if isinstance(height,tf.Tensor) and height.dtype != boxes.dtype:
            height = tf.cast(height, boxes.dtype)
        if isinstance(width,tf.Tensor) and width.dtype != boxes.dtype:
            width = tf.cast(width, boxes.dtype)
        ymin = boxes[...,0] / (height-1)
        xmin = boxes[...,1] / (width-1)
        ymax = boxes[...,2] / (height-1)
        xmax = boxes[...,3] / (width-1)

        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

def tfbatch_absolutely_boxes_to_relative_boxes(boxes, width, height):
    with tf.name_scope("batch_absolutely_boxes_to_relative_boxes"):
        batch_size,N,box_dim = btf.combined_static_and_dynamic_shape(boxes)
        boxes = tf.reshape(boxes,[-1,box_dim])
        res = tfabsolutely_boxes_to_relative_boxes(boxes,width,height)
        res = tf.reshape(res,[batch_size,N,box_dim])
        return res


def bboxes_and_labels_filter(bboxes,labels,lens,filter):
    with tf.name_scope("bboxes_and_labels_filter"):
        batch_size,nr,_ = btf.combined_static_and_dynamic_shape(bboxes)
        def fn(boxes,label,len):
           boxes = boxes[:len,:]
           label = label[:len]
           boxes,label = filter(boxes,label)
           len = tf.shape(label)[0]
           boxes = tf.pad(boxes,[[0,nr-len],[0,0]])
           label = tf.pad(label,[[0,nr-len]])
           return [boxes,label,len]
        return btf.static_or_dynamic_map_fn(lambda x:fn(x[0],x[1],x[2]),[bboxes,labels,lens],back_prop=False)
'''
box0:[ymin,xmin,ymax,xmax]
box1:[N,4],[ymin,xmin,ymax,xmax]
用box0裁box1,也就是说box1仅取与box0有重叠的部分
'''
def cut_boxes_by_box0(box0,box1):
    if not isinstance(box1,np.ndarray):
        box1 = np.array(box1)
    ymin,xmin,ymax,xmax = np.transpose(box1)
    ymin = np.minimum(np.maximum(box0[0],ymin),box0[2])
    xmin = np.minimum(np.maximum(box0[1],xmin),box0[3])
    ymax = np.maximum(np.minimum(box0[2],ymax),box0[0])
    xmax = np.maximum(np.minimum(box0[3],xmax),box0[1])
    box1 = np.stack([ymin,xmin,ymax,xmax],axis=0)
    return np.transpose(box1)

'''
bboxes:[N,4]
'''
def bboxes_rot90(bboxes,clockwise=True):
    bboxes = tf.transpose(bboxes)
    ymin,xmin,ymax,xmax = tf.unstack(bboxes,axis=0)
    if clockwise:
        nxmax = 1.0-xmin
        nxmin = 1.0-xmax
        bboxes = tf.stack([nxmin,ymin,nxmax,ymax],axis=0)
    else:
        nymax = 1.0-ymin
        nymin = 1.0-ymax
        bboxes = tf.stack([xmin,nymin,xmax,nymax],axis=0)
    bboxes = tf.transpose(bboxes)
    return bboxes

'''
bboxes:[N,4]
'''
def bboxes_rot90_ktimes(bboxes,clockwise=True,k=1):
    i = tf.constant(0)
    c = lambda i,box:tf.less(i,k)
    b = lambda i,box:(i+1,bboxes_rot90(box,clockwise))
    _,box = tf.while_loop(c,b,loop_vars=[i,bboxes])
    return box

'''
bboxes:[N,4]
'''
def bboxes_flip_up_down(bboxes):
    bboxes = tf.transpose(bboxes)
    ymin,xmin,ymax,xmax = tf.unstack(bboxes,axis=0)
    nymax = 1.0-ymin
    nymin = 1.0- ymax
    bboxes = tf.stack([nymin,xmin,nymax,xmax],axis=0)
    bboxes = tf.transpose(bboxes)
    return bboxes

'''
bboxes:[N,4]
'''
def bboxes_flip_left_right(bboxes):
    bboxes = tf.transpose(bboxes)
    ymin,xmin,ymax,xmax = tf.unstack(bboxes,axis=0)
    nxmax = 1.0-xmin
    nxmin = 1.0-xmax
    bboxes = tf.stack([ymin,nxmin,ymax,nxmax],axis=0)
    bboxes = tf.transpose(bboxes)
    return bboxes


'''
boxes0:[X,4]
boxes1:[Y,4]
boxes0与boxes1可以为同一个tensor
return:
dis:[X,Y] i,j表示boxes0[i]与boxes1[j]的交集与boxes1[j]的比值
'''
@btf.add_name_scope
def get_bboxes_intersection_matrix(boxes0,boxes1):
    X,_ = btf.combined_static_and_dynamic_shape(boxes0)
    Y,_ = btf.combined_static_and_dynamic_shape(boxes1)
    boxes0 = tf.expand_dims(boxes0,axis=1)
    boxes0 = tf.tile(boxes0,[1,Y,1])
    boxes1 = tf.expand_dims(boxes1,axis=0)
    boxes1 = tf.tile(boxes1,[X,1,1])

    boxes0 = tf.reshape(boxes0,[-1,4])
    boxes1 = tf.reshape(boxes1,[-1,4])
    scores = bboxes_intersection(boxes0,boxes1)
    return tf.reshape(scores,[X,Y])
'''
boxes0:[X,4]
boxes1:[Y,4]
return:
iou:[X,Y] i,j表示boxes0[i]与boxes1[j]的IOU
'''
@btf.add_name_scope
def get_iou_matrix(boxes0,boxes1):
    X,_ = btf.combined_static_and_dynamic_shape(boxes0)
    Y,_ = btf.combined_static_and_dynamic_shape(boxes1)
    boxes0 = tf.expand_dims(boxes0,axis=1)
    boxes0 = tf.tile(boxes0,[1,Y,1])
    boxes1 = tf.expand_dims(boxes1,axis=0)
    boxes1 = tf.tile(boxes1,[X,1,1])

    boxes0 = tf.reshape(boxes0,[-1,4])
    boxes1 = tf.reshape(boxes1,[-1,4])
    ious = batch_bboxes_jaccard(boxes0,boxes1)
    return tf.reshape(ious,[X,Y])

'''
boxes0:[X,4]
boxes1:[Y,4]
return:
dis:[X,Y] i,j表示boxes0[i]与boxes1[j]的l2距离
'''
@btf.add_name_scope
def get_bboxes_dis(boxes0,boxes1):
    X,_ = btf.combined_static_and_dynamic_shape(boxes0)
    Y,_ = btf.combined_static_and_dynamic_shape(boxes1)
    boxes0 = tf.expand_dims(boxes0,axis=1)
    boxes0 = tf.tile(boxes0,[1,Y,1])
    boxes1 = tf.expand_dims(boxes1,axis=0)
    boxes1 = tf.tile(boxes1,[X,1,1])

    boxes0 = tf.reshape(boxes0,[-1,4])
    boxes1 = tf.reshape(boxes1,[-1,4])
    boxes0 = get_bboxes_center_point(boxes0)
    boxes1 = get_bboxes_center_point(boxes1)
    dis = tf.square(boxes0-boxes1)
    dis = tf.reduce_sum(dis,axis=-1,keepdims=False)
    dis = tf.sqrt(dis)
    return tf.reshape(dis,[X,Y])

'''
boxes0:[X,4]
boxes1:[Y,4]
return:
is_in_center:[X,Y] i,j表示boxes1[j]的中心点是否在boxes0[i]内
'''
@btf.add_name_scope
def is_center_in_boxes(boxes0,boxes1):
    X,_ = btf.combined_static_and_dynamic_shape(boxes0)
    Y,_ = btf.combined_static_and_dynamic_shape(boxes1)
    boxes0 = tf.expand_dims(boxes0,axis=1)
    boxes0 = tf.tile(boxes0,[1,Y,1])
    boxes1 = tf.expand_dims(boxes1,axis=0)
    boxes1 = tf.tile(boxes1,[X,1,1])

    boxes0 = tf.reshape(boxes0,[-1,4])
    boxes1 = tf.reshape(boxes1,[-1,4])
    boxes1 = get_bboxes_center_point(boxes1)
    d0 = tf.greater_equal(boxes1[:,0],boxes0[:,0])
    d1 = tf.less_equal(boxes1[:,0],boxes0[:,2])
    d2 = tf.greater_equal(boxes1[:,1],boxes0[:,1])
    d3 = tf.less_equal(boxes1[:,1],boxes0[:,3])
    res = tf.logical_and(tf.logical_and(d0,d1),tf.logical_and(d2,d3))
    return tf.reshape(res,[X,Y])

'''
bboxes0: [B,X,4]
bboxes1: [B,Y,4]
len0: [B]
'''
def batch_bboxes_pair_wrap(bboxes0,bboxes1,fn,len0=None,len1=None,dtype=None):

    def fn0(tbboxes0,tbboxes1):
        res = fn(tbboxes0,tbboxes1)
        return res

    def fn1(tbboxes0,tbboxes1,l0):
        X,_ = btf.combined_static_and_dynamic_shape(tbboxes0)
        res = fn(tbboxes0[:l0],tbboxes1)
        res = tf.pad(res,[[0,X-l0],[0,0]])
        return res

    def fn2(tbboxes0,tbboxes1,l1):
        Y,_ = btf.combined_static_and_dynamic_shape(tbboxes1)
        res = fn(tbboxes0,tbboxes1[:l1])
        res = tf.pad(res,[[0,0],[0,Y-l1]])
        return res

    def fn3(tbboxes0,tbboxes1,l0,l1):
        X,_ = btf.combined_static_and_dynamic_shape(tbboxes0)
        Y,_ = btf.combined_static_and_dynamic_shape(tbboxes1)
        res = fn(tbboxes0[:l0],tbboxes1[:l1])
        res = tf.pad(res,[[0,X-l0],[0,Y-l1]])
        return res

    if dtype is None:
        dtype = bboxes0.dtype

    if len0 is None and len1 is None:
        return tf.map_fn(lambda x:fn0(x[0],x[1]),elems=(bboxes0,bboxes1),dtype=dtype)
    elif len0 is not None and len1 is not None:
        return tf.map_fn(lambda x:fn3(x[0],x[1],x[2],x[3]),elems=(bboxes0,bboxes1,len0,len1),dtype=dtype)
    elif len0 is not None and len1 is None:
        return tf.map_fn(lambda x:fn1(x[0],x[1],x[2]),elems=(bboxes0,bboxes1,len0),dtype=dtype)
    elif len0 is None and len1 is not None:
        return tf.map_fn(lambda x:fn2(x[0],x[1],x[2]),elems=(bboxes0,bboxes1,len1),dtype=dtype)

'''
bboxes0: [B,X,4]
bboxes1: [1,Y,4]
len0: [B]
'''
def batch_bboxes_pair_wrapv2(bboxes0,bboxes1,fn,len0=None,dtype=None,scope=None):

    def mfn(tbboxes0,l0):
        X,_ = btf.combined_static_and_dynamic_shape(tbboxes0)
        res = fn(tbboxes0[:l0],bboxes1[0])
        res = tf.pad(res,[[0,X-l0],[0,0]])
        return res

    if dtype is None:
        dtype = bboxes0.dtype

    with tf.name_scope(scope,default_name="batch_bboxes_pair_wrapv2"):
        return tf.map_fn(lambda x:mfn(x[0],x[1]),elems=(bboxes0,len0),dtype=dtype)

'''
用于删除同类别且有重叠的bboxes
如果box[i],与box[j]的交叉面积占box[j]面积的百分比大于threshold则删除box[j]
这种处理倾向于删除面积小的boxes
bboxes:[N,4] relative coordinate or absolute coordinate
labels:[N]
return:
bboxes:[Y,4],keep_pos:[N]
'''
@btf.add_name_scope
def remove_bboxes_by_overlap(bboxes,labels=None,threshold=0.5):
    scores = get_bboxes_intersection_matrix(bboxes,bboxes)
    R,_ = btf.combined_static_and_dynamic_shape(scores)
    scores = scores*(1.0-tf.eye(R))
    scores_t = tf.transpose(scores)
    scores_x = tf.reshape(tf.cast(tf.range(R*R),tf.float32),[R,R])
    scores_xt = tf.transpose(scores_x)
    faild_pos0 = tf.logical_and(tf.greater(scores,threshold),tf.less_equal(scores_t,threshold))
    faild_pos1 = tf.logical_and(tf.logical_and(tf.greater(scores,threshold),tf.greater(scores_t,threshold)),tf.greater(scores,scores_t))
    faild_pos2 = tf.logical_and(tf.logical_and(tf.greater(scores,threshold),tf.greater(scores_t,threshold)),tf.equal(scores,scores_t))
    faild_pos2 = tf.logical_and(tf.greater(scores_x,scores_xt),faild_pos2)
    faild_pos = tf.logical_or(faild_pos0,faild_pos1)
    faild_pos = tf.logical_or(faild_pos,faild_pos2)
    if labels is not None:
        labels0 = tf.reshape(labels,[R,1])
        labels0 = tf.tile(labels0,[1,R])
        labels1 = tf.reshape(labels,[1,R])
        labels1 = tf.tile(labels1,[R,1])
        test_pos = tf.equal(labels0,labels1)
        faild_pos = tf.logical_and(faild_pos,test_pos)

    faild_pos = tf.reduce_any(faild_pos,axis=0,keepdims=False)
    keep_pos = tf.logical_not(faild_pos)
    bboxes = tf.boolean_mask(bboxes,keep_pos)
    return bboxes,keep_pos

'''
用于删除同类别且有重叠的bboxes
如果box[i],与box[j]的交叉面积占box[j]面积的百分比大于threshold则删除box[i]
这种处理倾向于删除面积大的boxes
bboxes:[N,4] relative coordinate or absolute coordinate
labels:[N]
return:
bboxes:[Y,4],keep_pos:[N]
'''
@btf.add_name_scope
def remove_bboxes_by_overlapv2(bboxes,labels=None,threshold=0.5):
    scores = get_bboxes_intersection_matrix(bboxes,bboxes)
    R,_ = btf.combined_static_and_dynamic_shape(scores)
    scores = scores*(1.0-tf.eye(R))
    scores_t = tf.transpose(scores)
    scores_x = tf.reshape(tf.cast(tf.range(R*R),tf.float32),[R,R])
    scores_xt = tf.transpose(scores_x)
    faild_pos0 = tf.logical_and(tf.greater(scores,threshold),tf.less_equal(scores_t,threshold))
    faild_pos1 = tf.logical_and(tf.logical_and(tf.greater(scores,threshold),tf.greater(scores_t,threshold)),tf.greater(scores,scores_t))
    faild_pos2 = tf.logical_and(tf.logical_and(tf.greater(scores,threshold),tf.greater(scores_t,threshold)),tf.equal(scores,scores_t))
    faild_pos2 = tf.logical_and(tf.greater(scores_x,scores_xt),faild_pos2)
    faild_pos = tf.logical_or(faild_pos0,faild_pos1)
    faild_pos = tf.logical_or(faild_pos,faild_pos2)
    if labels is not None:
        labels0 = tf.reshape(labels,[R,1])
        labels0 = tf.tile(labels0,[1,R])
        labels1 = tf.reshape(labels,[1,R])
        labels1 = tf.tile(labels1,[R,1])
        test_pos = tf.equal(labels0,labels1)
        faild_pos = tf.logical_and(faild_pos,test_pos)

    faild_pos = tf.reduce_any(faild_pos,axis=1,keepdims=False)
    keep_pos = tf.logical_not(faild_pos)
    bboxes = tf.boolean_mask(bboxes,keep_pos)
    return bboxes,keep_pos

def batched_remove_bboxes_by_overlap(bboxes,labels=None,length=None,threshold=0.5):
    def fn0(boxes,label,l):
        nr,_ = btf.combined_static_and_dynamic_shape(boxes)
        boxes = boxes[:l,:]
        label = label[:l]
        boxes,keep_pos = remove_bboxes_by_overlap(boxes,label,threshold)
        n_nr = btf.combined_static_and_dynamic_shape(keep_pos)[0]
        padding_nr = nr-n_nr
        keep_pos = tf.pad(keep_pos,[[0,padding_nr]])
        n_nr,_ = btf.combined_static_and_dynamic_shape(boxes)
        padding_nr = nr-n_nr
        boxes = tf.pad(boxes,[[0,padding_nr],[0,0]])
        return boxes,keep_pos,n_nr

    def fn1(boxes,l):
        nr,_ = btf.combined_static_and_dynamic_shape(boxes)
        boxes = boxes[:l,:]
        boxes,keep_pos = remove_bboxes_by_overlap(boxes,None,threshold)
        n_nr,_ = btf.combined_static_and_dynamic_shape(boxes)
        padding_nr = nr-n_nr
        boxes = tf.pad(boxes,[[0,padding_nr],[0,0]])
        keep_pos = tf.pad(keep_pos,[[0,padding_nr]])
        return boxes,keep_pos,n_nr

    length = tf.convert_to_tensor(length)
    labels = tf.convert_to_tensor(labels)
    bboxes = tf.convert_to_tensor(bboxes)
    if labels is None:
        boxes,keep_pos,length = tf.map_fn(lambda x:fn1(x[0],x[1]),elems=(bboxes,length),
                          back_prop=False,dtype=(tf.float32,tf.bool,length.dtype))
    else:
        boxes,keep_pos,length = tf.map_fn(lambda x:fn0(x[0],x[1],x[2]),elems=(bboxes,labels,length),
                          back_prop=False,dtype=(tf.float32,tf.bool,length.dtype))

    return boxes,keep_pos,length

def change_bboxes_nr(bboxes0,labels0,bboxes1,labels1,threshold=0.8):
    if not isinstance(labels0,np.ndarray):
        labels0 = np.array(labels0)
    if not isinstance(labels1,np.ndarray):
        labels1 = np.array(labels1)
    nr = labels0.shape[0]
    same_ids = 0
    for i in range(nr):
        box0 = np.array([bboxes0[i]])
        ious = npbboxes_jaccard(box0,bboxes1)
        index = np.argmax(ious)
        if ious[index]>threshold and labels0[i] == labels1[index]:
            same_ids += 1
    return labels0.shape[0]+labels1.shape[0]-2*same_ids

