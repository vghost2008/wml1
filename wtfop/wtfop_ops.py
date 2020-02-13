#coding=utf-8
import tensorflow as tf
from tensorflow.python.framework import ops
import os
import numpy as np
from collections import Iterable

ops.NotDifferentiable("BoxesNms")
ops.NotDifferentiable("EBoxesNms")
ops.NotDifferentiable("DistoredBoxes")
ops.NotDifferentiable("RandomDistoredBoxes")
ops.NotDifferentiable("ProbabilityAdjust")
ops.NotDifferentiable("BoxesEncode1")
ops.NotDifferentiable("BoxesNmsNr2")
ops.NotDifferentiable("BoxesEncode")
ops.NotDifferentiable("BoxesMatch")
ops.NotDifferentiable("SetValue")
ops.NotDifferentiable("SparseMaskToDense")
ops.NotDifferentiable("PositionEmbedding")
ops.NotDifferentiable("BoxesSoftNms")
ops.NotDifferentiable("LabelType")
ops.NotDifferentiable("IntHash")
ops.NotDifferentiable("AnchorGenerator")
ops.NotDifferentiable("AdjacentMatrixGenerator")
ops.NotDifferentiable("BoxesMatchWithPred")
ops.NotDifferentiable("SampleLabels")
ops.NotDifferentiable("GetBoxesDeltas")

module_path = os.path.realpath(__file__)
module_dir = os.path.dirname(module_path)
lib_path = os.path.join(module_dir, 'wtfop.so')
wtfop_module = tf.load_op_library(lib_path)

def roi_pooling(input, rois, pool_height, pool_width,spatial_scale=1.0):
    out = wtfop_module.roi_pooling(input, rois, pool_height=pool_height, pool_width=pool_width,spatial_scale=spatial_scale)
    output, argmax_output = out[0], out[1]
    return output

def boxes_match(boxes, gboxes, glabels, glens,threshold=0.7):
    if glabels.dtype is not tf.int32:
        glabels = tf.cast(glabels,tf.int32)
    if glens.dtype is not tf.int32:
        glens = tf.cast(glens,tf.int32)
    out = wtfop_module.boxes_match(boxes=boxes, gboxes=gboxes, glabels=glabels, glens=glens,threshold=threshold)
    return out[0],out[1]

def boxes_match_with_pred(boxes, plabels,gboxes, glabels, glens,threshold=0.7):
    if glabels.dtype is not tf.int32:
        glabels = tf.cast(glabels,tf.int32)
    if plabels.dtype is not tf.int32:
        plabels = tf.cast(plabels,tf.int32)
    if glens.dtype is not tf.int32:
        glens = tf.cast(glens,tf.int32)
    out = wtfop_module.boxes_match_with_pred(boxes=boxes, plabels=plabels,gboxes=gboxes, glabels=glabels, glens=glens,threshold=threshold)
    return out[0],out[1]

def boxes_match_with_pred2(boxes, plabels,gboxes, glabels, glens,threshold=0.7,prio_scaling=[0.1,0.1,0.2,0.2]):
    if glabels.dtype is not tf.int32:
        glabels = tf.cast(glabels,tf.int32)
    if plabels.dtype is not tf.int32:
        plabels = tf.cast(plabels,tf.int32)
    if glens.dtype is not tf.int32:
        glens = tf.cast(glens,tf.int32)
    out = wtfop_module.boxes_match_with_pred2(boxes=boxes, plabels=plabels,gboxes=gboxes, glabels=glabels, glens=glens,threshold=threshold,prio_scaling=prio_scaling)
    return out[0],out[1],out[2]

def teeth_adjacent_matrix(boxes, labels, min_nr=8, min_dis=0.3):
    if labels.dtype is not tf.int32:
        labels = tf.cast(labels,tf.int32)
    out = wtfop_module.teeth_adjacent_matrix(boxes=boxes,  labels=labels,min_nr=min_nr,min_dis=min_dis);
    return out

def crop_boxes(ref_box, boxes, threshold=1.0):
    out = wtfop_module.crop_boxes(ref_box,boxes,threshold=threshold)
    return out[0],out[1]

def probability_adjust(probs,classes=[]):
    out = wtfop_module.probability_adjust(probs=probs,classes=classes)
    return out

def teeth_diseased_proc(teeth_boxes,diseased_boxes,diseased_labels,diseased_probability):
    out = wtfop_module.teeth_diseased_proc(teeth_boxes=teeth_boxes,diseased_boxes=diseased_boxes,diseased_labels=diseased_labels,diseased_probability=diseased_probability)
    return out

def slide_batch(data,filter,strides=None,padding="VALID"):
    if strides is None:
        strides = filter.get_shape().as_list()[:2]
    if data.get_shape().ndims == 2:
        data = tf.expand_dims(data,axis=2)
    if filter.get_shape().ndims == 2:
        filter = tf.expand_dims(filter,axis=2)
    out = wtfop_module.slide_batch(data=data,filter=filter,strides=strides,padding=padding)
    return out

def boxes_nms_nr(bboxes, classes, confidence=None,k=128,max_loop=5,classes_wise=True):
    if classes.dtype != tf.int32:
        classes = tf.cast(classes,tf.int32)
    out = wtfop_module.boxes_nms_nr(bottom_box=bboxes,classes=classes,k=k,max_loop=max_loop,classes_wise=classes_wise)
    return out[0],out[1],tf.cast(out[2],tf.int32)

def merge_character(bboxes, labels,dlabels, expand=0.01,super_box_type=68,space_type=69):
    out = wtfop_module.merge_character(bboxes=bboxes,labels=labels,dlabels=dlabels,expand=expand,super_box_type=super_box_type,space_type=space_type)
    return out
def simple_merge_character(labels,windex):
    out = wtfop_module.simple_merge_character(labels=labels,windex=windex)
    return out

def mach_words(targets, texts):
    out = wtfop_module.mach_words(targets=targets,texts=texts)
    return out

@ops.RegisterGradient("BoxesNmsNr")
def _boxes_nms_nr_grad(op, grad, _,_0):
  return [None,None]

def boxes_nms_nr2(bboxes, classes, k=128,threshold=0.8,classes_wise=True,confidence=None):
    if classes.dtype != tf.int32:
        classes = tf.cast(classes,tf.int32)
    out = wtfop_module.boxes_nms_nr2(bottom_box=bboxes,classes=classes,confidence=confidence,k=k,threshold=threshold,classes_wise=classes_wise)
    return out[0],out[1],tf.cast(out[2],tf.int32)


def wpad(tensor, padding):
    out = wtfop_module.w_pad(tensor=tensor,padding=padding)
    return out

@ops.RegisterGradient("WPad")
def _w_pad_grad(op, grad, _,_0):
  return [None,None]

def distored_boxes(boxes, scale=[],xoffset=[],yoffset=[],keep_org=True):
    out = wtfop_module.distored_boxes(boxes=boxes,scale=scale,xoffset=xoffset,yoffset=yoffset,keep_org=keep_org)
    return out
def random_distored_boxes(boxes, limits=[1.,1.,1.],size=4,keep_org=True):
    out = wtfop_module.random_distored_boxes(boxes=boxes,limits=limits,size=size,keep_org=keep_org)
    return out

def teeth_boxes_proc(boxes, probs, labels):
    out = wtfop_module.teeth_boxes_proc(boxes=boxes,probs=probs,labels=labels);
    return out[0],out[1],out[2],out[3],out[4],out[5]

def clean_teeth_boxes(boxes):
    out = wtfop_module.clean_teeth_boxes(boxes=boxes);
    return out[0],out[1]

def remove_boundary_boxes(size, boxes, threshold=0.9):
    out = wtfop_module.remove_boundary_boxes(size,boxes,threshold=threshold)
    return out[0],out[1]

def distored_qa(question,answer,expand_nr=2):
    out = wtfop_module.distored_qa(question,answer,expand_nr=expand_nr)
    return out[0],out[1],out[2]
def expand_tensor(tensor,expand_nr=2):
    out = wtfop_module.expand_tensor(tensor,expand_nr=expand_nr)
    return out
def boxes_nms(bboxes, classes, threshold=0.45,confidence=None,classes_wise=True,k=-1):
    if classes.dtype != tf.int32:
        classes = tf.cast(classes,tf.int32)
    out = wtfop_module.boxes_nms(bottom_box=bboxes,classes=classes,threshold=threshold,classes_wise=classes_wise,k=-1)
    return out[0],out[1],tf.cast(out[2],tf.int32)

def no_overlap_boxes_nms(bboxes, classes, threshold0=0.2,threshold1=0.8,confidence=None,classes_wise=True):
    if classes.dtype != tf.int32:
        classes = tf.cast(classes,tf.int32)
    out = wtfop_module.no_overlap_boxes_nms(bottom_box=bboxes,classes=classes,threshold0=threshold0,threshold1=threshold1,classes_wise=classes_wise)
    return out[0],out[1],tf.cast(out[2],tf.int32)

def group_boxes_nms(bboxes, classes, group,threshold=0.45,confidence=None):
    if classes.dtype != tf.int32:
        classes = tf.cast(classes,tf.int32)
    out = wtfop_module.group_boxes_nms(bottom_box=bboxes,classes=classes,group=group,threshold=threshold)
    return out[0],out[1],tf.cast(out[2],tf.int32)

def boxes_soft_nms(bboxes, classes, confidence,threshold=0.45,delta=0.5,classes_wise=True):
    if classes.dtype != tf.int32:
        classes = tf.cast(classes,tf.int32)
    out = wtfop_module.boxes_soft_nms(bottom_box=bboxes,classes=classes,confidence=confidence,threshold=threshold,delta=delta,classes_wise=classes_wise)
    return out[0],out[1],tf.cast(out[2],tf.int32)

def e_boxes_nms(bboxes, classes, threshold=0.45,k=0):
    if classes.dtype != tf.int32:
        classes = tf.cast(classes,tf.int32)
    out = wtfop_module.e_boxes_nms(bottom_box=bboxes,classes=classes,threshold=threshold,k=k)
    return out[0],out[1],tf.cast(out[2],tf.int32)

def boxes_encode(bboxes, gboxes,glabels,length,pos_threshold=0.7,neg_threshold=0.3,prio_scaling=[0.1,0.1,0.2,0.2],max_overlap_as_pos=True):
    if glabels.dtype != tf.int32:
        glabels= tf.cast(glabels,tf.int32)
    if bboxes.get_shape().ndims != 3:
        bboxes = tf.expand_dims(bboxes,axis=0)
    out = wtfop_module.boxes_encode(bottom_boxes=bboxes,bottom_gboxes=gboxes,bottom_glength=length,bottom_glabels=glabels,
    pos_threshold=pos_threshold,neg_threshold=neg_threshold,prio_scaling=prio_scaling,max_overlap_as_pos=max_overlap_as_pos)
    return out[0],out[1],out[2],out[3],out[4]

def matcher(bboxes, gboxes,glabels,length,pos_threshold=0.7,neg_threshold=0.3,max_overlap_as_pos=True):
    if glabels.dtype != tf.int32:
        glabels= tf.cast(glabels,tf.int32)
    if bboxes.get_shape().ndims != 3:
        bboxes = tf.expand_dims(bboxes,axis=0)
    out = wtfop_module.matcher(bottom_boxes=bboxes,bottom_gboxes=gboxes,bottom_glength=length,bottom_glabels=glabels,
    pos_threshold=pos_threshold,neg_threshold=neg_threshold,max_overlap_as_pos=max_overlap_as_pos)
    return out[0],out[1],out[2]

def get_boxes_deltas(boxes, gboxes,labels,indices,scale_weights=[10,10,5,5]):
    if labels.dtype != tf.int32:
        labels= tf.cast(labels,tf.int32)
    if boxes.get_shape().ndims != 3:
        boxes = tf.expand_dims(boxes,axis=0)
    out = wtfop_module.get_boxes_deltas(boxes=boxes,gboxes=gboxes,labels=labels,indices=indices,
    scale_weights=scale_weights)
    return out

def center_boxes_encode(gbboxes, glabels,glength,output_size,num_classes=2,max_box_nr=32,gaussian_iou=0.7):
    if glabels.dtype != tf.int32:
        glabels= tf.cast(glabels,tf.int32)
    out = wtfop_module.center_boxes_encode(gbboxes=gbboxes,glabels=glabels,glength=glength,
    output_size=output_size,num_classes=num_classes,max_box_nr=max_box_nr,gaussian_iou=gaussian_iou)
    return out[0],out[1],out[2],out[3],out[4]

def center_boxes_decode(heatmaps_tl,heatmaps_br,heatmaps_c,offset_tl,offset_br,offset_c,k=100):
    out = wtfop_module.center_boxes_decode(heatmaps_tl=heatmaps_tl,heatmaps_br=heatmaps_br,heatmaps_c=heatmaps_c,offset_tl=offset_tl,offset_br=offset_br,offset_c=offset_c,k=k)
    return out[0],out[1],out[2],out[3],out[4]

'''
 * prio_scaling:[4]
 * bottom_boxes:[X,4](ymin,xmin,ymax,xmax) 候选box,相对坐标
 * bottom_gboxes:[Y,4](ymin,xmin,ymax,xmax)ground truth box相对坐标
 * bottom_glabels:[Y] 0为背景
 * output_boxes:[X,4] regs(cy,cx,h,w)
 * output_labels:[X], 当前anchorbox的标签，背景为0,不为背景时为相应最大jaccard得分
 * output_scores:[X], 当前anchorbox与groundtruthbox的jaccard得分，当jaccard得分高于threshold时就不为背影
 * remove_indices:[X], 当前box是否要删除（不是正样本，也不是负样本)
'''
def boxes_encode1(bboxes, gboxes,glabels,pos_threshold=0.7,neg_threshold=0.3,prio_scaling=[0.1,0.1,0.2,0.2],max_overlap_as_pos=False):
    if glabels.dtype != tf.int32:
        glabels= tf.cast(glabels,tf.int32)
    out = wtfop_module.boxes_encode1(bottom_boxes=bboxes,bottom_gboxes=gboxes,bottom_glabels=glabels,
    pos_threshold=pos_threshold,neg_threshold=neg_threshold,prio_scaling=prio_scaling,max_overlap_as_pos=max_overlap_as_pos)
    return out[0],out[1],out[2],out[3]
def decode_boxes1(boxes,regs,prio_scaling=[0.1,0.1,0.2,0.2]):
    out = wtfop_module.decode_boxes1(bottom_boxes=boxes,bottom_regs=regs,prio_scaling=prio_scaling)
    return out 
def boxes_relative_to_absolute(boxes,width,height):
    out = wtfop_module.boxes_relative_to_absolute(bottom_boxes=boxes,height=height,width=width)
    #out = tf.cast(out,tf.int32)
    return out 
def boxes_select(bboxes, predictions,threshold=0.5,ignore_first=True):
    p_shape = predictions.shape.as_list()
    batch_size = p_shape[0] if len(p_shape) == 5 else 1
    predictions= tf.reshape(predictions, [batch_size, -1, p_shape[-1]])
    l_shape = bboxes.shape.as_list()
    bboxes= tf.reshape(bboxes, [batch_size, -1, l_shape[-1]])
    out = wtfop_module.boxes_select(bottom_box=bboxes,bottom_pred=predictions,threshold=threshold,ignore_first=ignore_first)
    return out[0],out[1],out[2],out[3]
def set_to_zero(input):
    out = wtfop_module.set_to_zero(input)
    return out 

def draw_points(image,points,color,point_size=1):
    out = wtfop_module.draw_points(image=image,points=points,color=color,point_size=point_size)
    return out 

def position_embedding(size):
    out = wtfop_module.position_embedding(size=size)
    if isinstance(size,list):
        out = tf.reshape(out,[1]+size)
    return out 

def plane_position_embedding(size):
    out = wtfop_module.plane_position_embedding(size=size)
    if isinstance(size,list):
        out = tf.reshape(out,[1]+size)
    return out 

def set_value(tensor,v,index):
    if index.get_shape().ndims is None or index.get_shape().ndims==0:
        index = tf.reshape(index,[-1])
    if index.dtype is not tf.int32:
        index = tf.cast(index,tf.int32)
    out = wtfop_module.set_value(tensor=tensor,v=v,index=index)
    return out 

def sparse_mask_to_dense(mask,labels,num_classes,set_background=True):
    if labels.dtype is not tf.int32:
        labels = tf.cast(labels,tf.int32)
    out = wtfop_module.sparse_mask_to_dense(mask=mask,labels=labels,num_classes=num_classes,set_background=set_background)
    return out 

def anchor_generator(shape,size,scales,aspect_ratios):
    if isinstance(scales,np.ndarray):
        scales = scales.tolist()
    if isinstance(aspect_ratios,np.ndarray):
        aspect_ratios= aspect_ratios.tolist()
    if not isinstance(size,tf.Tensor):
        size = tf.convert_to_tensor(size)
    if size.dtype != tf.float32:
        size = tf.cast(size,tf.float32)
    res = wtfop_module.anchor_generator(shape=shape,size=size,scales=scales,aspect_ratios=aspect_ratios)
    if not isinstance(shape,tf.Tensor):
        data_nr = len(aspect_ratios)*len(scales)*shape[0]*shape[1]
        res = tf.reshape(res,[data_nr,4])
    return res

def multi_anchor_generator(shape,size,scales,aspect_ratios):
    if isinstance(scales,np.ndarray):
        scales = scales.tolist()
    if isinstance(aspect_ratios,np.ndarray):
        aspect_ratios= aspect_ratios.tolist()
    res = wtfop_module.multi_anchor_generator(shape=shape,size=size,scales=scales,aspect_ratios=aspect_ratios)
    if not isinstance(shape,tf.Tensor):
        data_nr = len(aspect_ratios)*shape[0]*shape[1]
        res = tf.reshape(res,[data_nr,4])
    return res

def line_anchor_generator(shape,size,scales):
    if isinstance(scales,np.ndarray):
        scales = scales.tolist()
    if isinstance(size,tf.Tensor) and size.dtype != tf.float32:
        size = tf.cast(size,tf.float32)
    res = wtfop_module.line_anchor_generator(shape=shape,size=size,scales=scales)
    if not isinstance(shape,tf.Tensor):
        data_nr = len(scales)*shape[0]*shape[1]
        res = tf.reshape(res,[data_nr,4])
    return res

def random_range(max,hint,phy_max):
    if max.dtype is not tf.int32:
        max = tf.cast(max,tf.int32)
    if hint.dtype is not tf.int32:
        hint = tf.cast(hint,tf.int32)
    out = wtfop_module.random_range(max=max,hint=hint,phy_max=phy_max)
    return out[0],out[1] 

def int_hash(input,table):
    out = wtfop_module.int_hash(input=input,key=list(table.keys()),value=list(table.values()))
    return out

def adjacent_matrix_generator(bboxes,theta=100,scale=1.0,coord_scales=[1.0,1.0,1.0]):
    return wtfop_module.adjacent_matrix_generator(bboxes=bboxes,theta=theta,scale=scale,coord_scales=coord_scales)

def adjacent_matrix_generator_by_iou(bboxes,threshold=0.3,keep_connect=True):
    return wtfop_module.adjacent_matrix_generator_by_iou(bboxes=bboxes,threshold=threshold,keep_connect=keep_connect)

def mask_line_bboxes(mask,labels,lens,max_output_nr=-1):
    if mask.dtype != tf.uint8:
        mask = tf.cast(mask,tf.uint8)
    return wtfop_module.mask_line_bboxes(mask=mask,labels=labels,lens=lens,max_output_nr=max_output_nr)

def sample_labels(labels,ids,sample_nr=1024):
    return wtfop_module.sample_labels(labels=labels,ids=ids,sample_nr=sample_nr)

def merge_line_boxes(data,labels,bboxes,lens,threshold=0.5,dis_threshold=[0.1,0.1]):
    return wtfop_module.merge_line_boxes(data=data,labels=labels,bboxes=bboxes,lens=lens,threshold=threshold,dis_threshold=dis_threshold)

def get_image_resize_size(size,limit,align=1):
    if isinstance(limit,int):
        limit = [0,limit]
    if not isinstance(limit,tf.Tensor):
        limit = tf.convert_to_tesor(limit)
    if isinstance(align,int):
        align = [align,align]
    if not isinstance(align,tf.Tensor):
        align = tf.convert_to_tesor(align)
    return wtfop_module.get_image_resize_size(size=size,limit=limit,align=align)

@ops.RegisterGradient("RoiPooling")
def _roi_pool_grad(op, grad, _):
  data = op.inputs[0]
  rois = op.inputs[1]
  argmax = op.outputs[1]
  pooled_height = op.get_attr('pool_height')
  pooled_width = op.get_attr('pool_width')
  spatial_scale = op.get_attr('spatial_scale')

  data_grad = wtfop_module.roi_pooling_grad(data, rois, argmax, grad, pooled_height, pooled_width, spatial_scale)

  return [data_grad, None]

def median_blur(image,ksize=5):
    return wtfop_module.median_blur(image=image,ksize=ksize)

def bilateral_filter(image,d=5,sigmaColor=5,sigmaSpace=4):
    return wtfop_module.bilateral_filter(image=image,d=d,sigmaColor=sigmaColor,sigmaSpace=sigmaSpace)
