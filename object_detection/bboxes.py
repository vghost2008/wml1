#coding=utf-8
import numpy as np
import tensorflow as tf
import os
import sys
import random
sys.path.append(os.path.dirname(__file__))
import math
import object_detection.wmath as wmath
import wtfop.wtfop_ops as wtfop
import img_utils as wmli
import cv2 as cv
import copy

'''
sizes:如果is_area=True, sizes相对于整个原始图像的面积大小, 也就是说无论比例是多少，整个图的大小永远是1, 否则sizes为相对边的大小
ratios:高与宽的比例，实际计算时还需要考虑图像的形状，才能保持真正的比例
shape:[h,w]
返回的shape为[-1,4],表示每个位置，每个大小,每个比率的bboxes
也就是[位置0大小0比率0_anthorbox,位置0大小0比率1_anchorbox,...]
'''
def get_anchor_bboxes(shape=[38,38],sizes=[0.1,0.2],ratios=[1.,2.],is_area=False):
    anchor_bboxes = []
    HEIGHT = shape[0]
    WIDTH = shape[1]
    if not is_area:
        sizes = [s*s for s in sizes]
    for s in sizes:
        for a in ratios:
            '''
            s_h:为相对于HEIGHT的大小，s_w为相对于WIDTH的大小，如果要让实际比例保持不变，需要
            s_h*HEIGHT与s_w*WIDTH与实际比例一致
            '''
            s_h = math.sqrt(WIDTH/HEIGHT)*math.sqrt(s)
            s_w = math.sqrt(HEIGHT/WIDTH)*math.sqrt(s)

            bboxes = get_single_anchor_bboxes(shape,[s_h,s_w],a)
            bboxes = np.reshape(bboxes,[-1,4])
            anchor_bboxes.append(bboxes)
    res_bboxes = np.stack(anchor_bboxes,axis=1)
    res_bboxes = np.reshape(res_bboxes,[-1,4])
    return res_bboxes

'''
与原版本相比，不考虑featuremap 的比例
size:如果is_area为True size为面积大小,否则为边的大小
'''
def get_anchor_bboxesv2(shape=[38,38],sizes=[0.1,0.2],ratios=[1.,2.],is_area=False):
    anchor_bboxes = []
    if not is_area:
        sizes = [s*s for s in sizes]
    for s in sizes:
        for a in ratios:
            bboxes = get_single_anchor_bboxesv2(shape,s,a)
            bboxes = np.reshape(bboxes,[-1,4])
            anchor_bboxes.append(bboxes)
    res_bboxes = np.stack(anchor_bboxes,axis=1)
    res_bboxes = np.reshape(res_bboxes,[-1,4])
    return res_bboxes

'''
sizes:如果is_area=True, sizes相对于整个原始图像的面积大小, 也就是说无论比例是多少，整个图的大小永远是1, 否则sizes为相对边的大小
ratios:高与宽的比例，实际计算时还需要考虑图像的形状，才能保持真正的比例
shape:[h,w]
返回的shape为[-1,4],表示每个位置，,每个比率的bboxes,每个大小的anchorbox
与v1的差别为两点：
1, 先是大小后比率
2, 不nchor box对进行截断也就是说可以为负数或大于的
3，比率会在输入后进行翻转
进行上述操作主要是为了与tensorflow进行适配
也就是[位置0大小0比率0_anthorbox,位置0大小1比率0_anchorbox,...]
'''
def get_anchor_bboxesv3(shape=[38,38],sizes=[0.1,0.2],ratios=[1.,2.],is_area=False):
    anchor_bboxes = []
    HEIGHT = shape[0]
    WIDTH = shape[1]
    ratios = copy.deepcopy(ratios)
    ratios.reverse()
    if not is_area:
        sizes = [s*s for s in sizes]
    for a in ratios:
        for s in sizes:
            '''
            s_h:为相对于HEIGHT的大小，s_w为相对于WIDTH的大小，如果要让实际比例保持不变，需要
            s_h*HEIGHT与s_w*WIDTH与实际比例一致
            '''
            s_h = math.sqrt(WIDTH/HEIGHT)*math.sqrt(s)
            s_w = math.sqrt(HEIGHT/WIDTH)*math.sqrt(s)

            bboxes = get_single_anchor_bboxes(shape,[s_h,s_w],a,clamp=False)
            bboxes = np.reshape(bboxes,[-1,4])
            anchor_bboxes.append(bboxes)
    res_bboxes = np.stack(anchor_bboxes,axis=1)
    res_bboxes = np.reshape(res_bboxes,[-1,4])
    return res_bboxes
'''
输入：
size:表示相对大小[h_size,w_size],取两个值用于处理图像宽高比不相同的情况
ratio:表示高宽比

返回为一个np.array
返回的shape为shape+[4]
最后一维的四个数依次为ymin,xmin,ymax,xmax(相对坐标)
'''
def get_single_anchor_bboxes(shape=[38,38],size=[0.1,0.1],ratio=1.,clamp=True):
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    y_offset = 0.5/float(shape[0])
    x_offset = 0.5 / float(shape[1])
    if(shape[0]>1):
        y_step = (1.0-2.*y_offset)/float(shape[0]-1)
    else:
        y_step = 0.
    if(shape[1]>1):
        x_step = (1.0-2.*x_offset)/float(shape[1]-1)
    else:
        x_step = 0.
    y = y.astype(np.float32)* y_step+y_offset
    x = x.astype(np.float32) *x_step+x_offset

    h = size[0]*math.sqrt(ratio)
    w = size[1]/math.sqrt(ratio)
    h = np.ones(shape,dtype=np.float32)*h
    w = np.ones(shape,dtype=np.float32)*w

    y_min = y-h/2.
    x_min = x-w/2.
    y_max = y_min+h
    x_max = x_min+w

    res_data = np.stack([y_min,x_min,y_max,x_max],axis=2)
    if clamp:
        return correct_yxminmax_boxes(res_data)
    else:
        return res_data

def get_single_anchor_bboxesv2(shape=[38,38],size=0.1,ratio=1.):
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    y_offset = 0.5/float(shape[0])
    x_offset = 0.5 / float(shape[1])
    if(shape[0]>1):
        y_step = (1.0-2.*y_offset)/float(shape[0]-1)
    else:
        y_step = 0.
    if(shape[1]>1):
        x_step = (1.0-2.*x_offset)/float(shape[1]-1)
    else:
        x_step = 0.
    y = y.astype(np.float32)* y_step+y_offset
    x = x.astype(np.float32) *x_step+x_offset

    sqrt_size = math.sqrt(size)
    h = sqrt_size*math.sqrt(ratio)
    w = sqrt_size/math.sqrt(ratio)
    h = np.ones(shape,dtype=np.float32)*h
    w = np.ones(shape,dtype=np.float32)*w

    y_min = y-h/2.
    x_min = x-w/2.
    y_max = y_min+h
    x_max = x_min+w

    res_data = np.stack([y_min,x_min,y_max,x_max],axis=2)
    return correct_yxminmax_boxes(res_data)


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
                          scope=None):

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
将以ymin,xmin,ymax,xmax表示的box转换为以cy,cx,h,w表示的box
对data的shape没有限制
'''
def to_cxyhw(data):
    old_shape = tf.shape(data)
    data = tf.reshape(data,[-1,4])
    data = tf.transpose(data,perm=[1,0])
    ymin,xmin,ymax,xmax = tf.unstack(data,axis=0)
    cy = (ymin+ymax)/2.
    cx = (xmin+xmax)/2.
    h = ymax-ymin
    w = xmax-xmin
    data = tf.stack([cy,cx,h,w],axis=0)
    data = tf.transpose(data,perm=[1,0])
    data = tf.reshape(data,old_shape)
    return data
'''
将以cy,cx,h,w表示的box转换为以ymin,xmin,ymax,xmax表示的box
'''
def to_yxminmax(data):
    old_shape = tf.shape(data)
    data = tf.reshape(data,[-1,4])
    data = tf.transpose(data,perm=[1,0])
    cy, cx, h, w = tf.unstack(data,axis=0)
    ymin = cy-h/2.
    ymax = cy+h/2.
    xmin = cx-w/2.
    xmax = cx+w/2.
    data = tf.stack([ymin,xmin,ymax,xmax],axis=0)
    data = tf.transpose(data,perm=[1,0])
    data = tf.reshape(data,old_shape)

    return data

'''
input:[ymin,xmin,ymax,xmax]
output:[xmin,ymin,width,height]
'''
def to_xyminwh(bbox):
    return (bbox[1],bbox[0],bbox[3]-bbox[1]+1,bbox[2]-bbox[0]+1)

def scale_bboxes(bboxes,scale):
    old_shape = tf.shape(bboxes)
    data = tf.reshape(bboxes,[-1,4])
    data = tf.transpose(data,perm=[1,0])
    ymin,xmin,ymax,xmax = tf.unstack(data,axis=0)
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
    data = tf.stack([ymin, xmin, ymax, xmax], axis=0)
    data = tf.transpose(data, perm=[1, 0])
    data = tf.reshape(data, old_shape)
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

def random_distored_boxes(bboxes,limits=[0.,0.,0.],size=1,keep_org=True):
    if bboxes.get_shape().ndims == 2:
        return wtfop.random_distored_boxes(boxes=bboxes,limits=limits,size=size,keep_org=keep_org)
    else:
        return tf.map_fn(lambda x: wtfop.random_distored_boxes(boxes=x,limits=limits,size=size,keep_org=keep_org),
                         elems=bboxes,back_prop=False)
'''
box：[N,4] 使用相对坐标表
sub_box:bboxes参考区域
函数返回box在[0,0,1,1]区域的表示值
'''
def restore_sub_area(bboxes,sub_box):
    h = sub_box[2]-sub_box[0]
    w = sub_box[3]-sub_box[1]
    bboxes = tf.transpose(bboxes,[1,0])
    ymin,xmin,ymax,xmax = tf.unstack(bboxes,axis=0)
    ymin = ymin*h+sub_box[0]
    ymax = ymax*h+sub_box[0]
    xmin = xmin*w+sub_box[1]
    xmax = xmax*w+sub_box[1]
    bboxes = tf.stack([ymin,xmin,ymax,xmax],axis=0)
    bboxes = tf.transpose(bboxes)
    return bboxes

'''
boxes:[N,4],ymin,xmin,ymax,xmax
regs:[N,4]
'''
def decode_boxes(boxes,
                    regs,
                    prio_scaling=[0.1, 0.1, 0.2, 0.2]):

    l_shape = tf.shape(boxes)
    r_shape = tf.shape(regs)


    data = tf.transpose(boxes, perm=[1, 0])
    ymin, xmin, ymax, xmax = tf.unstack(data, axis=0)
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
        cy = regs[:, 0] * h * prio_scaling[0] + cy
        cx = regs[:, 1] * w * prio_scaling[1] + cx
        h = h * tf.exp(regs[:, 2] * prio_scaling[2])
        w = w * tf.exp(regs[:, 3] * prio_scaling[3])

    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin,xmin,ymax,xmax],axis=0)
    bboxes = tf.transpose(bboxes,perm=[1,0])
    bboxes = tf.reshape(bboxes, l_shape)
    return bboxes

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
    cuted_img = wmli.sub_image(segmentation,rect)
    contours,hierarchy = cv.findContours(cuted_img,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        boxes.append(bbox_of_contour(cnt))
    return contours,boxes

'''
bbox:(xmin,ymin,width,height)
'''
def random_int_in_bbox(bbox):
    x = random.randint(bbox[0],bbox[0]+bbox[2]-1)
    y = random.randint(bbox[1],bbox[1]+bbox[3]-1)
    return x,y

'''
bbox:(xmin,ymin,width,height)
size:(width,height) the size of return bbox
random return a box with center point in the input bbox
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
bbox:[(xmin,ymin,width,height),....]
return a list of new bbox with the size scale times of the input
'''
def expand_bbox(bboxes,scale=2):
    res_bboxes = []
    for bbox in bboxes:
        cx,cy = bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2
        new_width = bbox[2]*scale
        new_height = bbox[3]*scale
        min_x = cx-new_width//2
        min_y = cy-new_height//2
        res_bboxes.append((min_x,min_y,new_width,new_height))

    return res_bboxes
