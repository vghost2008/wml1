#coding=utf-8
import numpy as np
import math

def bboxes_jaccard(bboxes1, bboxes2):

    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)

    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w

    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    return jaccard
'''
box0,box1: ymin,xmin,ymax,xmax
'''
def box_jaccard(box0,box1):
    ymin0, xmin0, ymax0, xmax0 = box0[0],box0[1],box0[2],box0[3]
    ymin1, xmin1, ymax1, xmax1 = box1[0],box1[1],box1[2],box1[3]
    int_ymin = max(ymin0,ymin1)
    int_xmin = max(xmin0,xmin1)
    int_ymax = min(ymax0,ymax1)
    int_xmax = min(xmax0,xmax1)
    int_w = max(int_xmax-int_xmin,0.)
    int_h = max(int_ymax-int_ymin,0.)
    int_vol = int_w*int_h
    union_box_vol = box_vol(box0)+box_vol(box1)-int_vol

    if union_box_vol < 1e-8:
        return 0.0
    return int_vol/union_box_vol
'''
get the unio volume over bboxes's volume.
bboxes_ref:[4] ymin,xmin,ymax,xmax, relative coordinate
bboxes2:[X,4], ymin,xmin,ymax,xmax, relative coordinate
'''
def bboxes_intersection(bboxes_ref, bboxes):

    bboxes_ref = np.transpose(bboxes_ref)
    bboxes = np.transpose(bboxes)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes_ref[0], bboxes[0])
    int_xmin = np.maximum(bboxes_ref[1], bboxes[1])
    int_ymax = np.minimum(bboxes_ref[2], bboxes[2])
    int_xmax = np.minimum(bboxes_ref[3], bboxes[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
    score = int_vol / vol
    return score

'''
classes wise nms implementation by numpy.
classes:[X]
scores:[X]
bboxes:[X,4]
'''
def bboxes_nms(classes, scores, bboxes, nms_threshold=0.5):

    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_jaccard(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]

'''
crop a sub area sub_box in image, return the boxes IOU with sub_box greater than remove_threshold
bboxes:[X,4] ymin,xmin,ymax,xmax, relative coordinate
return:
bboxes:[Y,4],mask [X]
'''
def crop_box(bboxes,sub_box,remove_threshold=0.7):
    h = sub_box[2]-sub_box[0]
    w = sub_box[3]-sub_box[1]
    if h<1e-8 or w<1e-8:
        return None
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes)
    jaccard = bboxes_intersection(bboxes_ref=sub_box,bboxes=bboxes)
    mask = jaccard>remove_threshold
    bboxes = bboxes[mask]
    if bboxes.shape[0] == 0:
        return bboxes,mask
    top_left = np.array([[sub_box[0],sub_box[1],sub_box[0],sub_box[1]]],dtype=np.float32)
    bboxes = (bboxes-top_left)/np.array([[h,w,h,w]],dtype=np.float32)
    bboxes = correct_boxes(bboxes)
    return bboxes,mask

'''
bboxes:[X,4], ymin,xmin,ymax,xmax releative coordinate.
'''
def correct_boxes(bboxes):
    bboxes = np.transpose(bboxes,[1,0])
    ymin,xmin,ymax,xmax = bboxes[0],bboxes[1],bboxes[2],bboxes[3]
    ymin = np.maximum(ymin,0.)
    ymax = np.minimum(ymax,1.)
    xmin = np.maximum(xmin,0.)
    xmax = np.minimum(xmax,1.)
    bboxes = np.stack([ymin,xmin,ymax,xmax],axis=0)
    bboxes = np.transpose(bboxes)
    return bboxes

def box_vol(box):
    return (box[2]-box[0])*(box[3]-box[1])
'''
box0,box1: shape=[4] ymin,xmin,ymax,xmax
交叉面积占box0的百分比
'''
def bbox_intersection(box0,box1):
    ymin0, xmin0, ymax0, xmax0 = box0[0],box0[1],box0[2],box0[3]
    ymin1, xmin1, ymax1, xmax1 = box1[0],box1[1],box1[2],box1[3]
    int_ymin = max(ymin0,ymin1)
    int_xmin = max(xmin0,xmin1)
    int_ymax = min(ymax0,ymax1)
    int_xmax = min(xmax0,xmax1)
    int_w = max(int_xmax-int_xmin,0.)
    int_h = max(int_ymax-int_ymin,0.)
    int_vol = int_w*int_h
    box0_vol  = box_vol(box0)

    if box0_vol < 1e-8:
        return 0.0
    return int_vol/box0_vol

'''
删除与边界nms大于指定值的box
boundary:[h,w]
'''
def remove_boundary_boxes(bboxes,boundary=[0.1,0.1],threshold=0.9):
    assert len(bboxes.shape)==2,"error bboxes shape"
    boxes_nr = bboxes.shape[0]
    boundary_w = boundary[1]
    boundary_h = boundary[0]
    if boxes_nr == 0:
        return bboxes,np.array([],dtype=np.bool)
    keep_indicts = np.ones(shape=[boxes_nr],dtype=np.bool)
    boundary_boxes = np.array([[0.0,0.0,1.0,boundary_w],
                               [0.0, 1-boundary_w, 1.0, 1.0],
                              [0.0,0.0,boundary_h,1.0],
                               [1.0-boundary_h, 0.0, 1.0, 1.0]],dtype=np.float32)

    for i in range(boxes_nr):
        for j in range(boundary_boxes.shape[0]):
            if bbox_intersection(bboxes[i],boundary_boxes[j]) > threshold:
                keep_indicts[i] = False
                break

    return bboxes[keep_indicts],keep_indicts
'''
box：使用相对坐标表示，其参考区域为sub_box
函数返回box在[0,0,1,1]区域的表示值
'''
def restore_sub_area(bboxes,sub_box):
    h = sub_box[2]-sub_box[0]
    w = sub_box[3]-sub_box[1]
    bboxes = np.transpose(bboxes,[1,0])
    ymin,xmin,ymax,xmax = bboxes[0],bboxes[1],bboxes[2],bboxes[3]
    ymin = ymin*h+sub_box[0]
    ymax = ymax*h+sub_box[0]
    xmin = xmin*w+sub_box[1]
    xmax = xmax*w+sub_box[1]
    bboxes = np.stack([ymin,xmin,ymax,xmax],axis=0)
    bboxes = np.transpose(bboxes)
    return bboxes

def bboxes_flip_left_right(bboxes):
    bboxes = np.transpose(bboxes)
    ymin,xmin,ymax,xmax = bboxes[0],bboxes[1],bboxes[2],bboxes[3]
    nxmax = 1.0-xmin
    nxmin = 1.0-xmax
    bboxes = np.stack([ymin,nxmin,ymax,nxmax],axis=0)
    bboxes = np.transpose(bboxes)
    return bboxes

def bboxes_flip_up_down(bboxes):
    bboxes = np.transpose(bboxes)
    ymin,xmin,ymax,xmax = bboxes[0],bboxes[1],bboxes[2],bboxes[3]
    nymax = 1.0-ymin
    nymin = 1.0- ymax
    bboxes = np.stack([nymin,xmin,nymax,xmax],axis=0)
    bboxes = np.transpose(bboxes)
    return bboxes

'''
get a envolope box of boxes.
bboxes:[X,4]
'''
def envolope_of_boxes(bboxes):
    if bboxes.shape[0]==0:
        return np.array([0.,0.,1.,1.],dtype=np.float32)
    bboxes = np.transpose(bboxes,[1,0])
    ymin,xmin,ymax,xmax = bboxes[0],bboxes[1],bboxes[2],bboxes[3]
    ymin = ymin.min()
    xmin = xmin.min()
    ymax = ymax.max()
    xmax = xmax.max()

    return np.array([ymin,xmin,ymax,xmax])

def minmax_to_cyxhw(bboxes):
    bboxes = np.transpose(bboxes,[1,0])
    ymin,xmin,ymax,xmax = bboxes[0],bboxes[1],bboxes[2],bboxes[3]
    cx = (xmax+xmin)/2.
    cy = (ymax+ymin)/2.
    w = (xmax-xmin)
    h = (ymax-ymin)
    bboxes = np.stack([cy,cx,h,w],axis=0)
    bboxes = np.transpose(bboxes,axes=[1,0])
    return bboxes

def minmax_to_cyxsr(bboxes,h=1.0,w=1.0):
    bboxes = np.transpose(bboxes,[1,0])
    ymin,xmin,ymax,xmax = bboxes[0],bboxes[1],bboxes[2],bboxes[3]
    cx = (xmax+xmin)/2.
    cy = (ymax+ymin)/2.
    w = (xmax-xmin)*w
    h = (ymax-ymin)*h
    s = w*h
    r = h/w
    bboxes = np.stack([cy,cx,s,r],axis=0)
    bboxes = np.transpose(bboxes,axes=[1,0])
    return bboxes

def cyxhw_to_minmax(bboxes):
    bboxes = np.transpose(bboxes,[1,0])
    cy, cx, h, w = bboxes[0],bboxes[1],bboxes[2],bboxes[3]
    ymin = cy-h/2.
    ymax = cy+h/2.
    xmin = cx-w/2.
    xmax = cx+w/2.
    bboxes = np.stack([ymin,xmin,ymax,xmax],axis=0)
    bboxes = np.transpose(bboxes)
    return bboxes
'''
return the distance of two boxes's center point.
'''
def box_dis(box0,box1):
    box0 = minmax_to_cyxhw([box0])
    box1 = minmax_to_cyxhw([box1])

    dy = box0[0][0]-box1[0][0]
    dx = box0[0][1]-box1[0][1]
    return math.sqrt(dy*dy+dx*dx)
'''
return the aspect(h/w) of a box.
'''
def box_aspect(boxes):
    bboxes = minmax_to_cyxhw(boxes)
    bboxes = np.transpose(bboxes,[1,0])
    cy, cx, h, w = bboxes[0],bboxes[1],bboxes[2],bboxes[3]
    aspect = h/w
    return aspect

def bboxes_decode(feat_localizations,
                      anchor_bboxes,
                      prior_scaling=[0.1, 0.1, 0.2, 0.2]):

    l_shape = feat_localizations.shape
    feat_localizations = np.reshape(feat_localizations,
                                    (-1, l_shape[-2], l_shape[-1]))
    yref, xref, href, wref = anchor_bboxes
    xref = np.reshape(xref, [-1, 1])
    yref = np.reshape(yref, [-1, 1])

    cx = feat_localizations[:, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, 1] * href * prior_scaling[1] + yref
    w = wref * np.exp(feat_localizations[:, :, 2] * prior_scaling[2])
    h = href * np.exp(feat_localizations[:, :, 3] * prior_scaling[3])
    # bboxes: ymin, xmin, xmax, ymax.
    bboxes = np.zeros_like(feat_localizations)
    bboxes[:, :, 0] = cy - h / 2.
    bboxes[:, :, 1] = cx - w / 2.
    bboxes[:, :, 2] = cy + h / 2.
    bboxes[:, :, 3] = cx + w / 2.
    bboxes = np.reshape(bboxes, l_shape)
    return bboxes

def bboxes_selectv2(predictions_layer,
                            localizations_layer,
                            select_threshold=0.5):

    p_shape = predictions_layer.shape
    batch_size = p_shape[0] if len(p_shape) == 5 else 1
    predictions_layer = np.reshape(predictions_layer,
                                   (batch_size, -1, p_shape[-1]))
    l_shape = localizations_layer.shape
    localizations_layer = np.reshape(localizations_layer,
                                     (batch_size, -1, l_shape[-1]))

    sub_predictions = predictions_layer[:, :, 1:]
    idxes = np.where(sub_predictions > select_threshold)
    classes = idxes[-1] + 1
    scores = sub_predictions[idxes]
    bboxes = localizations_layer[idxes[:-1]]

    return classes, scores, bboxes

def bboxes_sort(classes, scores, bboxes, top_k=400):
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes
