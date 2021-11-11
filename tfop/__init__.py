#coding=utf-8
import tensorflow as tf
from tensorflow.python.framework import ops
import os
import numpy as np
from collections import Iterable
import traceback
import sys

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
ops.NotDifferentiable("BoxesMatchWithPred")
ops.NotDifferentiable("SampleLabels")
ops.NotDifferentiable("GetBoxesDeltas")
ops.NotDifferentiable("ItemAssign")
ops.NotDifferentiable("OpenPoseEncode")
ops.NotDifferentiable("OpenPoseDecode")
ops.NotDifferentiable("Center2BoxesEncode")
ops.NotDifferentiable("DmprPsDecoder")

module_path = os.path.realpath(__file__)
module_dir = os.path.dirname(module_path)
lib_path = os.path.join(module_dir, 'tfop.so')
print(lib_path)
tfop_module = tf.load_op_library(lib_path)
random_select = tfop_module.random_select
min_area_rect = tfop_module.min_area_rect
min_area_rect_with_bboxes  = tfop_module.min_area_rect_with_bboxes 
full_size_mask = tfop_module.full_size_mask
tensor_rotate = tfop_module.tensor_rotate
left_pool = tfop_module.left_pool
right_pool = tfop_module.right_pool
bottom_pool = tfop_module.bottom_pool
top_pool = tfop_module.top_pool
make_neg_pair_index = tfop_module.make_neg_pair_index
center_boxes_filter= tfop_module.center_boxes_filter
fcos_boxes_encode = tfop_module.fcos_boxes_encode
fill_bboxes = tfop_module.fill_b_boxes
his_random_select = tfop_module.his_random_select
item_assign= tfop_module.item_assign
qc_post_process = tfop_module.qc_post_process 
merge_instance_by_mask = tfop_module.merge_instance_by_mask 
adjacent_matrix_generator_by_iouv2 = tfop_module.adjacent_matrix_generator_by_iou_v2
adjacent_matrix_generator_by_iouv3 = tfop_module.adjacent_matrix_generator_by_iou_v3
adjacent_matrix_generator_by_iouv4 = tfop_module.adjacent_matrix_generator_by_iou_v4
counting = tfop_module.counting
cell_encode_label = tfop_module.cell_encode_label
cell_decode_label = tfop_module.cell_decode_label
cell_encode_label2 = tfop_module.cell_encode_label2
cell_decode_label2 = tfop_module.cell_decode_label2
bboxes_rotate = tfop_module.bboxes_rotate
counter = tfop_module.counter
#deform_conv_op = tfop_module.deform_conv_op
#deform_conv_grad_op = tfop_module.deform_conv_backprop_op

def mask_rotate(mask,angle,get_bboxes_stride=None):
    mask = tfop_module.mask_rotate(mask=mask,angle=angle)
    bbox = get_bboxes_from_mask(mask,stride=get_bboxes_stride)

    return mask,bbox

def get_bboxes_from_mask(mask,stride=None):
    if stride is None or stride==1:
        return tfop_module.get_bboxes_from_mask(mask=mask)
    else:
        size = tf.shape(mask)//stride
        mask = tf.expand_dims(mask,axis=-1)
        mask = tf.image.resize_nearest_neighbor(mask,size[1:3])
        mask = tf.squeeze(mask,axis=-1)
        return tfop_module.get_bboxes_from_mask(mask=mask)*stride

def roi_pooling(input, rois, pool_height, pool_width,spatial_scale=1.0):
    out = tfop_module.roi_pooling(input, rois, pool_height=pool_height, pool_width=pool_width,spatial_scale=spatial_scale)
    output, argmax_output = out[0], out[1]
    return output

def boxes_match(boxes, gboxes, glabels, glens,threshold=0.7):
    if glabels.dtype is not tf.int32:
        glabels = tf.cast(glabels,tf.int32)
    if glens.dtype is not tf.int32:
        glens = tf.cast(glens,tf.int32)
    out = tfop_module.boxes_match(boxes=boxes, gboxes=gboxes, glabels=glabels, glens=glens,threshold=threshold)
    return out[0],out[1],out[2]

def boxes_match_with_pred(boxes, plabels,pprobs,gboxes, glabels, glens,threshold=0.7,is_binary_plabels=False):
    if glabels.dtype is not tf.int32:
        glabels = tf.cast(glabels,tf.int32)
    if plabels.dtype is not tf.int32:
        plabels = tf.cast(plabels,tf.int32)
    if glens.dtype is not tf.int32:
        glens = tf.cast(glens,tf.int32)
    out = tfop_module.boxes_match_with_pred(boxes=boxes, plabels=plabels,pprobs=pprobs,gboxes=gboxes, glabels=glabels, glens=glens,threshold=threshold,is_binary_plabels=is_binary_plabels)
    return out[0],out[1],out[2]

def boxes_match_with_predv3(boxes, plabels,pprobs,gboxes, glabels, glens,threshold=0.7,is_binary_plabels=False,sort_by_probs=False):
    if glabels.dtype is not tf.int32:
        glabels = tf.cast(glabels,tf.int32)
    if plabels.dtype is not tf.int32:
        plabels = tf.cast(plabels,tf.int32)
    if glens.dtype is not tf.int32:
        glens = tf.cast(glens,tf.int32)
    out = tfop_module.boxes_match_with_pred_v3(boxes=boxes, plabels=plabels,pprobs=pprobs,gboxes=gboxes, glabels=glabels, glens=glens,threshold=threshold,is_binary_plabels=is_binary_plabels,
    sort_by_probs=sort_by_probs)
    return out[0],out[1],out[2]

def boxes_match_with_pred2(boxes, plabels,gboxes, glabels, glens,threshold=0.7,prio_scaling=[0.1,0.1,0.2,0.2]):
    if glabels.dtype is not tf.int32:
        glabels = tf.cast(glabels,tf.int32)
    if plabels.dtype is not tf.int32:
        plabels = tf.cast(plabels,tf.int32)
    if glens.dtype is not tf.int32:
        glens = tf.cast(glens,tf.int32)
    out = tfop_module.boxes_match_with_pred2(boxes=boxes, plabels=plabels,gboxes=gboxes, glabels=glabels, glens=glens,threshold=threshold,prio_scaling=prio_scaling)
    return out[0],out[1],out[2]

def teeth_adjacent_matrix(boxes, labels, min_nr=8, min_dis=0.3):
    if labels.dtype is not tf.int32:
        labels = tf.cast(labels,tf.int32)
    out = tfop_module.teeth_adjacent_matrix(boxes=boxes,  labels=labels,min_nr=min_nr,min_dis=min_dis);
    return out

def crop_boxes(ref_box, boxes, threshold=1.0):
    out = tfop_module.crop_boxes(ref_box,boxes,threshold=threshold)
    return out[0],out[1]

def probability_adjust(probs,classes=[]):
    out = tfop_module.probability_adjust(probs=probs,classes=classes)
    return out

def teeth_diseased_proc(teeth_boxes,diseased_boxes,diseased_labels,diseased_probability):
    out = tfop_module.teeth_diseased_proc(teeth_boxes=teeth_boxes,diseased_boxes=diseased_boxes,diseased_labels=diseased_labels,diseased_probability=diseased_probability)
    return out

def slide_batch(data,filter,strides=None,padding="VALID"):
    if strides is None:
        strides = filter.get_shape().as_list()[:2]
    if data.get_shape().ndims == 2:
        data = tf.expand_dims(data,axis=2)
    if filter.get_shape().ndims == 2:
        filter = tf.expand_dims(filter,axis=2)
    out = tfop_module.slide_batch(data=data,filter=filter,strides=strides,padding=padding)
    return out

def boxes_nms_nr(bboxes, classes, confidence=None,k=128,max_loop=5,classes_wise=True):
    if classes.dtype != tf.int32:
        classes = tf.cast(classes,tf.int32)
    out = tfop_module.boxes_nms_nr(bottom_box=bboxes,classes=classes,k=k,max_loop=max_loop,classes_wise=classes_wise)
    return out[0],out[1],tf.cast(out[2],tf.int32)

@ops.RegisterGradient("BoxesNmsNr")
def _boxes_nms_nr_grad(op, grad, _,_0):
  return [None,None]

def boxes_nms_nr2(bboxes, classes, k=128,threshold=0.8,classes_wise=True,confidence=None,fast_mode=False,allow_less_output=False):
    if classes.dtype != tf.int32:
        classes = tf.cast(classes,tf.int32)
    if confidence is None:
        confidence = tf.ones_like(classes,tf.float32)
    out = tfop_module.boxes_nms_nr2(bottom_box=bboxes,classes=classes,confidence=confidence,k=k,threshold=threshold,classes_wise=classes_wise,
                                    fast_mode=fast_mode,
                                    allow_less_output=allow_less_output)
    return out[0],out[1],tf.cast(out[2],tf.int32)


def wpad(tensor, padding):
    out = tfop_module.w_pad(tensor=tensor,padding=padding)
    return out

@ops.RegisterGradient("WPad")
def _w_pad_grad(op, grad, _,_0):
  return [None,None]

def distored_boxes(boxes, scale=[],xoffset=[],yoffset=[],keep_org=True):
    out = tfop_module.distored_boxes(boxes=boxes,scale=scale,xoffset=xoffset,yoffset=yoffset,keep_org=keep_org)
    return out
def random_distored_boxes(boxes, limits=[1.,1.,1.],size=4,keep_org=True):
    out = tfop_module.random_distored_boxes(boxes=boxes,limits=limits,size=size,keep_org=keep_org)
    return out

def teeth_boxes_proc(boxes, probs, labels):
    out = tfop_module.teeth_boxes_proc(boxes=boxes,probs=probs,labels=labels);
    return out[0],out[1],out[2],out[3],out[4],out[5]

def clean_teeth_boxes(boxes):
    out = tfop_module.clean_teeth_boxes(boxes=boxes);
    return out[0],out[1]

def remove_boundary_boxes(size, boxes, threshold=0.9):
    out = tfop_module.remove_boundary_boxes(size,boxes,threshold=threshold)
    return out[0],out[1]

def distored_qa(question,answer,expand_nr=2):
    out = tfop_module.distored_qa(question,answer,expand_nr=expand_nr)
    return out[0],out[1],out[2]
def expand_tensor(tensor,expand_nr=2):
    out = tfop_module.expand_tensor(tensor,expand_nr=expand_nr)
    return out
def boxes_nms(bboxes, classes, threshold=0.45,confidence=None,classes_wise=True,k=-1):
    if classes.dtype != tf.int32:
        classes = tf.cast(classes,tf.int32)
    out = tfop_module.boxes_nms(bottom_box=bboxes,classes=classes,threshold=threshold,classes_wise=classes_wise,k=k)
    return out[0],out[1],tf.cast(out[2],tf.int32)

def no_overlap_boxes_nms(bboxes, classes, threshold0=0.2,threshold1=0.8,confidence=None,classes_wise=True):
    if classes.dtype != tf.int32:
        classes = tf.cast(classes,tf.int32)
    out = tfop_module.no_overlap_boxes_nms(bottom_box=bboxes,classes=classes,threshold0=threshold0,threshold1=threshold1,classes_wise=classes_wise)
    return out[0],out[1],tf.cast(out[2],tf.int32)

def group_boxes_nms(bboxes, classes, group,threshold=0.45,confidence=None):
    if classes.dtype != tf.int32:
        classes = tf.cast(classes,tf.int32)
    out = tfop_module.group_boxes_nms(bottom_box=bboxes,classes=classes,group=group,threshold=threshold)
    return out[0],out[1],tf.cast(out[2],tf.int32)

def boxes_soft_nms(bboxes, classes, confidence,threshold=0.45,delta=0.5,classes_wise=True):
    if classes.dtype != tf.int32:
        classes = tf.cast(classes,tf.int32)
    out = tfop_module.boxes_soft_nms(bottom_box=bboxes,classes=classes,confidence=confidence,threshold=threshold,delta=delta,classes_wise=classes_wise)
    return out[0],out[1],tf.cast(out[2],tf.int32)

def e_boxes_nms(bboxes, classes, threshold=0.45,k=0):
    if classes.dtype != tf.int32:
        classes = tf.cast(classes,tf.int32)
    out = tfop_module.e_boxes_nms(bottom_box=bboxes,classes=classes,threshold=threshold,k=k)
    return out[0],out[1],tf.cast(out[2],tf.int32)

def boxes_encode(bboxes, gboxes,glabels,length,pos_threshold=0.7,neg_threshold=0.3,prio_scaling=[0.1,0.1,0.2,0.2],max_overlap_as_pos=True):
    if glabels.dtype != tf.int32:
        glabels= tf.cast(glabels,tf.int32)
    if bboxes.get_shape().ndims != 3:
        bboxes = tf.expand_dims(bboxes,axis=0)
    out = tfop_module.boxes_encode(bottom_boxes=bboxes,bottom_gboxes=gboxes,bottom_glength=length,bottom_glabels=glabels,
    pos_threshold=pos_threshold,neg_threshold=neg_threshold,prio_scaling=prio_scaling,max_overlap_as_pos=max_overlap_as_pos)
    return out[0],out[1],out[2],out[3],out[4]

def matcher(bboxes, gboxes,glabels,length,pos_threshold=0.7,neg_threshold=0.3,max_overlap_as_pos=True,force_in_gtbox=False):
    if glabels.dtype != tf.int32:
        glabels= tf.cast(glabels,tf.int32)
    if bboxes.get_shape().ndims != 3:
        bboxes = tf.expand_dims(bboxes,axis=0)
    out = tfop_module.matcher(bottom_boxes=bboxes,bottom_gboxes=gboxes,bottom_glength=length,bottom_glabels=glabels,
    pos_threshold=pos_threshold,neg_threshold=neg_threshold,max_overlap_as_pos=max_overlap_as_pos,force_in_gtbox=force_in_gtbox)
    return out[0],out[1],out[2]

def matcherv2(bboxes, gboxes,glabels,length,threshold=[0.3,0.3]):
    if glabels.dtype != tf.int32:
        glabels= tf.cast(glabels,tf.int32)
    if bboxes.get_shape().ndims != 3:
        bboxes = tf.expand_dims(bboxes,axis=0)
    out = tfop_module.matcher_v2(bottom_boxes=bboxes,bottom_gboxes=gboxes,bottom_glength=length,bottom_glabels=glabels,
    threshold=threshold)
    return out[0],out[1],out[2]

def get_boxes_deltas(boxes, gboxes,labels,indices,scale_weights=[10,10,5,5]):
    if labels.dtype != tf.int32:
        labels= tf.cast(labels,tf.int32)
    if boxes.get_shape().ndims != 3:
        boxes = tf.expand_dims(boxes,axis=0)
    out = tfop_module.get_boxes_deltas(boxes=boxes,gboxes=gboxes,labels=labels,indices=indices,
    scale_weights=scale_weights)
    return out

def center_boxes_encode(gbboxes, glabels,glength,output_size,num_classes=2,max_box_nr=32,gaussian_iou=0.7):
    if glabels.dtype != tf.int32:
        glabels= tf.cast(glabels,tf.int32)
    out = tfop_module.center_boxes_encode(gbboxes=gbboxes,glabels=glabels,glength=glength,
    output_size=output_size,num_classes=num_classes,max_box_nr=max_box_nr,gaussian_iou=gaussian_iou)
    return out[0],out[1],out[2],out[3],out[4]

def center_boxes_decode(heatmaps_tl,heatmaps_br,heatmaps_c,offset_tl,offset_br,offset_c,k=100):
    out = tfop_module.center_boxes_decode(heatmaps_tl=heatmaps_tl,heatmaps_br=heatmaps_br,heatmaps_c=heatmaps_c,offset_tl=offset_tl,offset_br=offset_br,offset_c=offset_c,k=k)
    return out[0],out[1],out[2],out[3],out[4]

def center2_boxes_encode(gbboxes, glabels,glength,output_size,num_classes=2,gaussian_iou=0.7):
    if glabels.dtype != tf.int32:
        glabels= tf.cast(glabels,tf.int32)
    out = tfop_module.center2_boxes_encode(gbboxes=gbboxes,glabels=glabels,glength=glength,
    output_size=output_size,num_classes=num_classes,gaussian_iou=gaussian_iou)
    c_map,hw_offset,mask = out[0],out[1],out[2]
    return c_map,hw_offset,mask

def center2_boxes_decode(heatmaps,offset,hw,k=100,threshold=0.1):
    out = tfop_module.center2_boxes_decode(heatmaps=heatmaps,offset=offset,hw=hw,k=k,threshold=threshold)
    bboxes,labels,probs,index,lens = out[0],out[1],out[2],out[3],out[4]

    return bboxes,labels,probs,index,lens


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
    out = tfop_module.boxes_encode1(bottom_boxes=bboxes,bottom_gboxes=gboxes,bottom_glabels=glabels,
    pos_threshold=pos_threshold,neg_threshold=neg_threshold,prio_scaling=prio_scaling,max_overlap_as_pos=max_overlap_as_pos)
    return out[0],out[1],out[2],out[3]
def decode_boxes1(boxes,regs,prio_scaling=[0.1,0.1,0.2,0.2]):
    out = tfop_module.decode_boxes1(bottom_boxes=boxes,bottom_regs=regs,prio_scaling=prio_scaling)
    return out 
def boxes_relative_to_absolute(boxes,width,height):
    out = tfop_module.boxes_relative_to_absolute(bottom_boxes=boxes,height=height,width=width)
    #out = tf.cast(out,tf.int32)
    return out 
def boxes_select(bboxes, predictions,threshold=0.5,ignore_first=True):
    p_shape = predictions.shape.as_list()
    batch_size = p_shape[0] if len(p_shape) == 5 else 1
    predictions= tf.reshape(predictions, [batch_size, -1, p_shape[-1]])
    l_shape = bboxes.shape.as_list()
    bboxes= tf.reshape(bboxes, [batch_size, -1, l_shape[-1]])
    out = tfop_module.boxes_select(bottom_box=bboxes,bottom_pred=predictions,threshold=threshold,ignore_first=ignore_first)
    return out[0],out[1],out[2],out[3]
def set_to_zero(input):
    out = tfop_module.set_to_zero(input)
    return out 

def draw_points(image,points,color,point_size=1):
    out = tfop_module.draw_points(image=image,points=points,color=color,point_size=point_size)
    return out 

def position_embedding(size):
    out = tfop_module.position_embedding(size=size)
    if isinstance(size,list):
        out = tf.reshape(out,[1]+size)
    return out 

def plane_position_embedding(size,ref_size=[512,512]):
    out = tfop_module.plane_position_embedding(size=size,ref_size=ref_size)
    if isinstance(size,Iterable) and not isinstance(size,tf.Tensor):
        out = tf.reshape(out,[1]+list(size))
    return out 

def set_value(tensor,v,index):
    if index.get_shape().ndims is None or index.get_shape().ndims<2:
        print(f"\n-------------------------------------------------------")
        print(f"WARNING reset index shape {traceback.format_exc()}")
        print(f"-------------------------------------------------------\n")
        index = tf.reshape(index,[1,-1])
    if index.dtype is not tf.int32:
        index = tf.cast(index,tf.int32)
    #tensor = tf.Print(tensor,["shape:",tf.shape(tensor),tf.shape(v),tf.shape(index)])
    out = tfop_module.set_value(tensor=tensor,v=v,index=index)
    return out 

def sparse_mask_to_dense(mask,labels,num_classes,set_background=True):
    if labels.dtype is not tf.int32:
        labels = tf.cast(labels,tf.int32)
    out = tfop_module.sparse_mask_to_dense(mask=mask,labels=labels,num_classes=num_classes,set_background=set_background)
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
    res = tfop_module.anchor_generator(shape=shape,size=size,scales=scales,aspect_ratios=aspect_ratios)
    if not isinstance(shape,tf.Tensor):
        data_nr = len(aspect_ratios)*len(scales)*shape[0]*shape[1]
        res = tf.reshape(res,[data_nr,4])
    return res

def multi_anchor_generator(shape,size,scales,aspect_ratios):
    if isinstance(scales,np.ndarray):
        scales = scales.tolist()
    if isinstance(aspect_ratios,np.ndarray):
        aspect_ratios= aspect_ratios.tolist()
    res = tfop_module.multi_anchor_generator(shape=shape,size=size,scales=scales,aspect_ratios=aspect_ratios)
    if not isinstance(shape,tf.Tensor):
        data_nr = len(aspect_ratios)*shape[0]*shape[1]
        res = tf.reshape(res,[data_nr,4])
    return res

def line_anchor_generator(shape,size,scales):
    if isinstance(scales,np.ndarray):
        scales = scales.tolist()
    if not isinstance(size,tf.Tensor):
        size = tf.convert_to_tensor(size)
    if size.dtype != tf.float32:
        size = tf.cast(size,tf.float32)
    res = tfop_module.line_anchor_generator(shape=shape,size=size,scales=scales)
    if not isinstance(shape,tf.Tensor):
        data_nr = len(scales)*shape[0]*shape[1]
        res = tf.reshape(res,[data_nr,4])
    return res

def random_range(max,hint,phy_max):
    if max.dtype is not tf.int32:
        max = tf.cast(max,tf.int32)
    if hint.dtype is not tf.int32:
        hint = tf.cast(hint,tf.int32)
    out = tfop_module.random_range(max=max,hint=hint,phy_max=phy_max)
    return out[0],out[1] 

def int_hash(input,table):
    out = tfop_module.int_hash(input=input,key=list(table.keys()),value=list(table.values()))
    return out

def adjacent_matrix_generator(bboxes,theta=100,scale=1.0,coord_scales=[1.0,1.0,1.0]):
    return tfop_module.adjacent_matrix_generator(bboxes=bboxes,theta=theta,scale=scale,coord_scales=coord_scales)

def adjacent_matrix_generator_by_iou(bboxes,threshold=0.3,keep_connect=True):
    return tfop_module.adjacent_matrix_generator_by_iou(bboxes=bboxes,threshold=threshold,keep_connect=keep_connect)

def mask_line_bboxes(mask,labels,lens,max_output_nr=-1):
    if mask.dtype != tf.uint8:
        mask = tf.cast(mask,tf.uint8)
    return tfop_module.mask_line_bboxes(mask=mask,labels=labels,lens=lens,max_output_nr=max_output_nr)

def sample_labels(labels,ids,line_no,sample_nr=1024):
    return tfop_module.sample_labels(labels=labels,ids=ids,line_no=line_no,sample_nr=sample_nr)

def merge_line_boxes(data,labels,bboxes,threshold=0.5,dis_threshold=[0.1,0.1]):
    return tfop_module.merge_line_boxes(data=data,labels=labels,bboxes=bboxes,threshold=threshold,dis_threshold=dis_threshold)

def get_image_resize_size(size,limit,align=1):
    if isinstance(limit,int):
        limit = [0,limit]
    if not isinstance(limit,tf.Tensor):
        limit = tf.convert_to_tensor(limit)
    if isinstance(align,int):
        align = [align,align]
    if not isinstance(align,tf.Tensor):
        align = tf.convert_to_tensor(align)
    return tfop_module.get_image_resize_size(size=size,limit=limit,align=align)

@ops.RegisterGradient("RoiPooling")
def _roi_pool_grad(op, grad, _):
  data = op.inputs[0]
  rois = op.inputs[1]
  argmax = op.outputs[1]
  pooled_height = op.get_attr('pool_height')
  pooled_width = op.get_attr('pool_width')
  spatial_scale = op.get_attr('spatial_scale')

  data_grad = tfop_module.roi_pooling_grad(data, rois, argmax, grad, pooled_height, pooled_width, spatial_scale)

  return [data_grad, None]

def median_blur(image,ksize=5):
    return tfop_module.median_blur(image=image,ksize=ksize)

def bilateral_filter(image,d=5,sigmaColor=5,sigmaSpace=4):
    return tfop_module.bilateral_filter(image=image,d=d,sigmaColor=sigmaColor,sigmaSpace=sigmaSpace)

def hr_net_encode(keypoints,output_size,glength,gaussian_delta=2.0):
    out = tfop_module.hr_net_pe(keypoints=keypoints,output_size=output_size,glength=glength,
    gaussian_delta=gaussian_delta)
    return out[0],out[1]

def match_by_tag(tag_k,loc_k,val_k,detection_threshold=0.1,tag_threshold=1.0,use_detection_val=True):
    out = tfop_module.match_by_tag(tag_k=tag_k,
                                    loc_k=loc_k,
                                    val_k=val_k,
                                    detection_threshold=detection_threshold,
                                    tag_threshold=tag_threshold,
                                    use_detection_val=use_detection_val)
    return out

def hr_net_refine(ans,det,tag):
    out = tfop_module.hr_net_refine(ans=ans,det=det,tag=tag)
    return out

def open_pose_encode(keypoints,output_size,glength,keypoints_pair,l_delta=2.0,gaussian_delta=2.0):
    out = tfop_module.open_pose_encode(keypoints=keypoints,output_size=output_size,glength=glength,
    keypoints_pair=keypoints_pair,l_delta=l_delta,gaussian_delta=gaussian_delta)
    return out

def open_pose_decode(conf_maps,paf_maps,keypoints_pair,keypoints_th=0.1,interp_samples=10,paf_score_th=0.1,conf_th=0.7,max_detection=100):
    out = tfop_module.open_pose_decode(conf_maps=conf_maps,
    paf_maps=paf_maps,
    keypoints_pair=keypoints_pair,
    keypoints_th=keypoints_th,
    interp_samples=interp_samples,
    paf_score_th=paf_score_th,
    conf_th=conf_th,
    max_detection=max_detection)
    return out

#OCR ops
def merge_character(bboxes, labels,dlabels, expand=0.01,super_box_type=68,space_type=69):
    out = tfop_module.merge_character(bboxes=bboxes,labels=labels,dlabels=dlabels,expand=expand,super_box_type=super_box_type,space_type=space_type)
    return out
def simple_merge_character(labels,windex):
    out = tfop_module.simple_merge_character(labels=labels,windex=windex)
    return out

def mach_words(targets, texts):
    out = tfop_module.mach_words(targets=targets,texts=texts)
    return out

def fair_mot(bboxes,probs,embedding,is_first_frame,det_thredh=0.1,frame_rate=30,track_buffer=30,assignment_thresh=[0.4,0.5,0.7],return_losted=False):
    out = tfop_module.fair_mot(bboxes=bboxes,probs=probs,embedding=embedding,is_first_frame=is_first_frame,det_thredh=det_thredh,
                                frame_rate=frame_rate,track_buffer=track_buffer,
                                assignment_thresh=assignment_thresh,
                                return_losted=return_losted)
    return out[0],out[1],out[2]

def sort_mot(bboxes,probs,is_first_frame,det_thredh=0.1,frame_rate=30,track_buffer=30,assignment_thresh=[0.4,0.5,0.7]):
    out = tfop_module.sort_mot(bboxes=bboxes,probs=probs,is_first_frame=is_first_frame,det_thredh=det_thredh,
                                frame_rate=frame_rate,track_buffer=track_buffer,
                                assignment_thresh=assignment_thresh)
    return out[0],out[1],out[2]

@ops.RegisterGradient("DeformConvOp")
def _deform_conv_grad(op, grad):
    """The gradients for `deform_conv`.
    Args:
      op: The `deform_conv` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
      grad: Gradient with respect to the output of the `roi_pool` op.
    Returns:
      Gradients with respect to the input of `zero_out`.
    """
    data = op.inputs[0]
    filter = op.inputs[1]
    offset = op.inputs[2]

    strides = op.get_attr('strides')
    rates = op.get_attr('rates')
    num_groups = op.get_attr('num_groups')
    padding = op.get_attr('padding')
    data_format = op.get_attr('data_format')
    deformable_group = op.get_attr('deformable_group')

    # compute gradient
    data_grad = deform_conv_grad_op(data, filter, offset, grad, strides, rates, num_groups, deformable_group, padding,
                                    data_format)
    return data_grad

@ops.RegisterGradient("LeftPool")
def _left_pool(op, grad):
  tensor = op.inputs[0]
  fw_output = op.outputs[0]

  data_grad = tfop_module.left_pool_grad(tensor,fw_output,grad)

  return data_grad

@ops.RegisterGradient("RightPool")
def _right_pool(op, grad):
  tensor = op.inputs[0]
  fw_output = op.outputs[0]

  data_grad = tfop_module.right_pool_grad(tensor,fw_output,grad)

  return data_grad

@ops.RegisterGradient("BottomPool")
def _bottom_pool(op, grad):
  tensor = op.inputs[0]
  fw_output = op.outputs[0]

  data_grad = tfop_module.bottom_pool_grad(tensor,fw_output,grad)

  return data_grad

@ops.RegisterGradient("TopPool")
def _top_pool(op, grad):
  tensor = op.inputs[0]
  fw_output = op.outputs[0]

  data_grad = tfop_module.top_pool_grad(tensor,fw_output,grad)

  return data_grad
