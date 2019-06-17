#coding=utf-8
import tensorflow as tf
import numpy as np
import wml_tfutils as wml
from wtfop.wtfop_ops import decode_boxes1
from wtfop.wtfop_ops import boxes_nms,boxes_nms_nr2,boxes_nms_nr,boxes_soft_nms
import wtfop.wtfop_ops as wop
import object_detection.bboxes as wml_bboxes
import img_utils
import wml_tfutils as wmlt
import object_detection.wlayers as wlayers
import time
slim = tf.contrib.slim

'''
only process batch_size equal one. 
class_prediction:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率
shape为[batch_size,X,num_classes]

bboxes_regs:模型预测的每个proposal_bboxes/anchro boxes/default boxes到目标真实bbox的回归参数
shape为[batch_size,X,4](classes_wise=Flase)或者(batch_size,X,num_classes,4](classes_wise=True)
proposal_bboxes:候选box
shape为[batch_size,X,4]
threshold:选择class_prediction的阀值
nms_threshold: nms阀值
candiate_nr:选择proposal_bboxes中预测最好的candiate_nr个bboxes进行筛选
limits:[4,2],用于对回归参数的范围进行限制，分别对应于cy,cx,h,w的回归参数，limits的值对应于prio_scaling=[1,1,1,1]是的设置
prio_scaling:在encode_boxes以及decode_boxes是使用的prio_scaling参数
返回:
boxes:[1,Y,4]
labels:[1,Y]
probability:[1,Y]
'''
def get_prediction(class_prediction,
                   bboxes_regs,
                   proposal_bboxes,
                   limits=None,
                   prio_scaling=[0.1,0.1,0.2,0.2],
                   threshold=0.5,
                   nms_threshold=0.1,
                   candiate_nr = 1500,
                   classes_wise=False,
                   classes_wise_nms=True):
    if class_prediction.get_shape().as_list()[0] is not None:
        batch_size = class_prediction.get_shape().as_list()[0]
        assert batch_size==1,"error batch size."
    else:
        a_op = tf.assert_equal(tf.shape(class_prediction)[0],1,data=[tf.shape(class_prediction)],name="assert_batch_size")
        with tf.control_dependencies([a_op]):
            class_prediction = tf.identity(class_prediction)
    #删除背景
    class_prediction = class_prediction[:,:,1:]
    probability,nb_labels = tf.nn.top_k(class_prediction,k=1)
    #背景的类别为0，前面已经删除了背景，需要重新加上
    labels = nb_labels+1
    ndims = class_prediction.get_shape().ndims
    probability = tf.squeeze(probability, axis=ndims - 1)
    labels = tf.squeeze(labels, axis=ndims - 1)
    res_indices = tf.range(tf.shape(labels)[1])*tf.ones_like(labels)

    if (isinstance(proposal_bboxes,tf.Tensor) and proposal_bboxes.shape.ndims<3) or (isinstance(proposal_bboxes,np.ndarray) and len(proposal_bboxes.shape)<3):
        proposal_bboxes = tf.expand_dims(proposal_bboxes,axis=0)

    #按类别在bboxes_regs选择相应类的回归参数
    if classes_wise:
        old_bboxes_regs_shape = bboxes_regs.get_shape().as_list()
        nb_labels = tf.reshape(nb_labels,[-1])
        bboxes_regs = tf.reshape(bboxes_regs,[-1,old_bboxes_regs_shape[-2],old_bboxes_regs_shape[-1]])
        bboxes_regs = wml.select_2thdata_by_index(bboxes_regs,nb_labels)
        bboxes_regs = wml.reshape(bboxes_regs,[old_bboxes_regs_shape[0],old_bboxes_regs_shape[1],old_bboxes_regs_shape[3]])
    '''
    NMS前数据必须已经排好序
    通过top_k+gather排序
    '''
    probability,indices = tf.nn.top_k(probability,k=tf.shape(probability)[1])
    labels = wml.gather_in_axis(labels,indices,axis=1)
    bboxes_regs = wml.gather_in_axis(bboxes_regs,indices,axis=1)
    proposal_bboxes = wml.gather_in_axis(proposal_bboxes,indices,axis=1)
    res_indices = wml.gather_in_axis(res_indices,indices,axis=1)

    pmask = tf.greater(probability,threshold)
    probability = tf.boolean_mask(probability,pmask)
    labels = tf.boolean_mask(labels,pmask)
    proposal_bboxes = tf.boolean_mask(proposal_bboxes,pmask)
    boxes_regs = tf.boolean_mask(bboxes_regs,pmask)
    res_indices = tf.boolean_mask(res_indices,pmask)
    if limits is not None:
        limits = np.array(limits)/np.array(zip(prio_scaling,prio_scaling))
        cy,cx,h,w = tf.unstack(tf.transpose(boxes_regs,perm=[1,0]),axis=0)
        cy = tf.clip_by_value(cy,clip_value_min=limits[0][0],clip_value_max=limits[0][1])
        cx = tf.clip_by_value(cx,clip_value_min=limits[1][0],clip_value_max=limits[1][1])
        h = tf.clip_by_value(h,clip_value_min=limits[2][0],clip_value_max=limits[2][1])
        w = tf.clip_by_value(w,clip_value_min=limits[3][0],clip_value_max=limits[3][1])
        boxes_regs = tf.stack([cy,cx,h,w],axis=0)
        boxes_regs = tf.transpose(boxes_regs)

    boxes = decode_boxes1(proposal_bboxes,boxes_regs)
    boxes,labels,indices = boxes_nms(boxes,labels,threshold=nms_threshold,classes_wise=classes_wise_nms)
    probability = tf.gather(probability,indices)
    res_indices = tf.gather(res_indices,indices)

    probability,indices = tf.nn.top_k(probability,k=tf.minimum(candiate_nr,tf.shape(probability)[0]))
    labels = tf.gather(labels,indices)
    boxes = tf.gather(boxes,indices)
    res_indices = tf.gather(res_indices,indices)
    boxes = tf.reshape(boxes,[1,-1,4])
    labels = tf.reshape(labels,[1,-1])
    probability = tf.reshape(probability,[1,-1])
    res_indices = tf.reshape(res_indices,[1,-1])
    return boxes,labels,probability,res_indices

'''
this version of get_prediction have no batch dim.

class_prediction:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率
shape为[X,num_classes]

bboxes_regs:模型预测的每个proposal_bboxes/anchro boxes/default boxes到目标真实bbox的回归参数
shape为[X,4](classes_wise=Flase)或者(X,num_classes,4](classes_wise=True)

proposal_bboxes:候选box
shape为[X,4]

threshold:选择class_prediction的阀值

nms_threshold: nms阀值

candiate_nr:选择proposal_bboxes中预测最好的candiate_nr个bboxes进行筛选

limits:[4,2],用于对回归参数的范围进行限制，分别对应于cy,cx,h,w的回归参数，limits的值对应于prio_scaling=[1,1,1,1]是的设置
prio_scaling:在encode_boxes以及decode_boxes是使用的prio_scaling参数

nms:nms函数,是否使用softnms,使用soft nms与不使用soft nms时, nms_threshold的意义有很大的区别， 不使用soft nms时，nms_threshold表示
IOU小于nms_threshold的两个bbox为不同目标，使用soft nms时，nms_threshold表示得分高于nms_threshold的才是真目标

返回:
boxes:[candiate_nr,4]
labels:[candiate_nr]
probability:[candiate_nr]
len: the available boxes number
'''
def __get_predictionv2(class_prediction,
                   bboxes_regs,
                   proposal_bboxes,
                   limits=None,
                   prio_scaling=[0.1,0.1,0.2,0.2],
                   threshold=0.5,
                   classes_wise=False,
                   candiate_nr = 1500,
                   nms=None):
    #删除背景
    class_prediction = class_prediction[:,1:]
    probability,nb_labels = tf.nn.top_k(class_prediction,k=1)
    #背景的类别为0，前面已经删除了背景，需要重新加上
    labels = nb_labels+1
    ndims = class_prediction.get_shape().ndims
    probability = tf.squeeze(probability, axis=ndims - 1)
    labels = tf.squeeze(labels, axis=ndims - 1)
    res_indices = tf.range(tf.shape(labels)[0])

    #按类别在bboxes_regs选择相应类的回归参数
    if classes_wise:
        nb_labels = tf.reshape(nb_labels,[-1])
        bboxes_regs = wml.select_2thdata_by_index(bboxes_regs,nb_labels)
    '''
    NMS前数据必须已经排好序
    通过top_k+gather排序
    '''
    probability,indices = tf.nn.top_k(probability,k=tf.shape(probability)[0])
    labels = wml.gather_in_axis(labels,indices,axis=0)
    bboxes_regs = wml.gather_in_axis(bboxes_regs,indices,axis=0)
    proposal_bboxes = wml.gather_in_axis(proposal_bboxes,indices,axis=0)
    res_indices = wml.gather_in_axis(res_indices,indices,axis=0)

    pmask = tf.greater(probability,threshold)
    probability = tf.boolean_mask(probability,pmask)
    labels = tf.boolean_mask(labels,pmask)
    proposal_bboxes = tf.boolean_mask(proposal_bboxes,pmask)
    boxes_regs = tf.boolean_mask(bboxes_regs,pmask)
    res_indices = tf.boolean_mask(res_indices,pmask)
    if limits is not None:
        limits = np.array(limits)/np.array(zip(prio_scaling,prio_scaling))
        cy,cx,h,w = tf.unstack(tf.transpose(boxes_regs,perm=[1,0]),axis=0)
        cy = tf.clip_by_value(cy,clip_value_min=limits[0][0],clip_value_max=limits[0][1])
        cx = tf.clip_by_value(cx,clip_value_min=limits[1][0],clip_value_max=limits[1][1])
        h = tf.clip_by_value(h,clip_value_min=limits[2][0],clip_value_max=limits[2][1])
        w = tf.clip_by_value(w,clip_value_min=limits[3][0],clip_value_max=limits[3][1])
        boxes_regs = tf.stack([cy,cx,h,w],axis=0)
        boxes_regs = tf.transpose(boxes_regs)

    boxes = decode_boxes1(proposal_bboxes,boxes_regs)
    #boxes,labels,indices = boxes_nms(boxes,labels,threshold=nms_threshold,classes_wise=classes_wise_nms)
    boxes,labels,indices = nms(boxes,labels,confidence=probability)
    probability = tf.gather(probability,indices)
    res_indices = tf.gather(res_indices,indices)

    probability,indices = tf.nn.top_k(probability,k=tf.minimum(candiate_nr,tf.shape(probability)[0]))
    labels = tf.gather(labels,indices)
    boxes = tf.gather(boxes,indices)
    res_indices = tf.gather(res_indices,indices)
    len = tf.shape(probability)[0]
    boxes = tf.pad(boxes,paddings=[[0,candiate_nr-len],[0,0]])
    labels = tf.pad(labels,paddings=[[0,candiate_nr-len]])
    probability = tf.pad(probability,paddings=[[0,candiate_nr-len]])
    res_indices = tf.pad(res_indices,paddings=[[0,candiate_nr-len]])
    boxes = tf.reshape(boxes,[candiate_nr,4])
    labels = tf.reshape(labels,[candiate_nr])
    probability = tf.reshape(probability,[candiate_nr])
    res_indices = tf.reshape(res_indices,[candiate_nr])
    return boxes,labels,probability,res_indices,len

'''
class_prediction:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率
shape为[batch_size,X,num_classes]

bboxes_regs:模型预测的每个proposal_bboxes/anchro boxes/default boxes到目标真实bbox的回归参数
shape为[batch_size,X,4](classes_wise=Flase)或者(batch_size,X,num_classes,4](classes_wise=True)

proposal_bboxes:候选box
shape为[batch_size,X,4] (ymin,xmin,ymax,xmax) relative coordinate

threshold:选择class_prediction的阀值

candiate_nr:选择proposal_bboxes中预测最好的candiate_nr个bboxes进行筛选

limits:[4,2],用于对回归参数的范围进行限制，分别对应于cy,cx,h,w的回归参数，limits的值对应于prio_scaling=[1,1,1,1]是的设置
prio_scaling:在encode_boxes以及decode_boxes是使用的prio_scaling参数

返回:
boxes:[batch_size,candiate_nr,4]
labels:[batch_size,candiate_nr]
probability:[batch_size,candiate_nr]
indices:[batch_size,candiate_nr]
len:[batch_size] the available boxes number
'''
def get_predictionv2(class_prediction,
                   bboxes_regs,
                   proposal_bboxes,
                   limits=None,
                   prio_scaling=[0.1,0.1,0.2,0.2],
                   threshold=0.5,
                   candiate_nr = 1500,
                   classes_wise=False,
                   nms=None):
    if proposal_bboxes.get_shape().as_list()[0] == 1:
        '''
        In single stage model, the proposal box are anchor boxes(or default boxes) and for any batch the anchor boxes is the same.
        '''
        proposal_bboxes = tf.squeeze(proposal_bboxes,axis=0)
        boxes,labels,probability,res_indices,lens = tf.map_fn(lambda x:__get_predictionv2(x[0],x[1],proposal_bboxes,limits,prio_scaling,
                                                                                          threshold,
                                                                                          candiate_nr=candiate_nr,
                                                                                          classes_wise=classes_wise,
                                                                                          nms=nms),
                                                              elems=(class_prediction,bboxes_regs),dtype=(tf.float32,tf.int32,tf.float32,tf.int32,tf.int32)
                                                              )
    else:
        boxes,labels,probability,res_indices,lens = tf.map_fn(lambda x:__get_predictionv2(x[0],x[1],x[2],limits,prio_scaling,
                                                                                          threshold,
                                                                                          candiate_nr=candiate_nr,
                                                                                          classes_wise=classes_wise,
                                                                                          nms=nms),
                                                              elems=(class_prediction,bboxes_regs,proposal_bboxes),dtype=(tf.float32,tf.int32,tf.float32,tf.int32,tf.int32)
                                                              )
    return boxes,labels,probability,res_indices,lens

'''
class_prediction:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率
shape为[batch_size,X,num_classes]

bboxes_regs:模型预测的每个proposal_bboxes/anchro boxes/default boxes到目标真实bbox的回归参数
shape为[batch_size,X,4](classes_wise=Flase)或者(batch_size,X,num_classes,4](classes_wise=True)

proposal_bboxes:候选box
shape为[batch_size,X,4]

threshold:选择class_prediction的阀值

nms_threshold: nms阀值

candiate_nr:选择proposal_bboxes中预测最好的candiate_nr个bboxes

limits:[4,2],用于对回归参数的范围进行限制，分别对应于cy,cx,h,w的回归参数，limits的值对应于prio_scaling=[1,1,1,1]是的设置
prio_scaling:在encode_boxes以及decode_boxes是使用的prio_scaling参数

返回:
boxes:[batch_size,candiate_nr,4]
labels:[batch_size,candiate_nr]
probability:[batch_size,candiate_nr]
'''
def get_predictionv3(class_prediction,
                   bboxes_regs,
                   proposal_bboxes,
                   limits=None,
                   prio_scaling=[0.1,0.1,0.2,0.2],
                   nms_threshold=0.1,
                   candiate_nr = 1500,
                   classes_wise=False,
                   classes_wise_nms=True):
    #删除背景
    class_prediction = class_prediction[:,:,1:]
    probability,nb_labels = tf.nn.top_k(class_prediction,k=1)
    #背景的类别为0，前面已经删除了背景，需要重新加上
    labels = nb_labels+1
    ndims = class_prediction.get_shape().ndims
    probability = tf.squeeze(probability, axis=ndims - 1)
    labels = tf.squeeze(labels, axis=ndims - 1)

    if (isinstance(proposal_bboxes,tf.Tensor) and proposal_bboxes.shape.ndims<3) or (isinstance(proposal_bboxes,np.ndarray) and len(proposal_bboxes.shape)<3):
        proposal_bboxes = tf.expand_dims(proposal_bboxes,axis=0)

    #按类别在bboxes_regs选择相应类的回归参数
    if classes_wise:
        old_bboxes_regs_shape = bboxes_regs.get_shape().as_list()
        nb_labels = tf.reshape(nb_labels,[-1])
        bboxes_regs = tf.reshape(bboxes_regs,[-1,old_bboxes_regs_shape[-2],old_bboxes_regs_shape[-1]])
        bboxes_regs = wml.select_2thdata_by_index(bboxes_regs,nb_labels)
        bboxes_regs = wml.reshape(bboxes_regs,[old_bboxes_regs_shape[0],old_bboxes_regs_shape[1],old_bboxes_regs_shape[3]])
    '''
    NMS前数据必须已经排好序
    通过top_k+gather排序
    '''
    probability,indices = tf.nn.top_k(probability,k=tf.shape(probability)[1])
    shape_2d = labels.get_shape().as_list()
    shape_box = bboxes_regs.get_shape().as_list()
    labels = wml.gather_in_axis(labels,indices,axis=1)
    bboxes_regs = wml.gather_in_axis(bboxes_regs,indices,axis=1)
    proposal_bboxes = wml.gather_in_axis(proposal_bboxes,indices,axis=1)

    labels = wml.reshape(labels,shape_2d)
    bboxes_regs = wml.reshape(bboxes_regs,shape_box)
    proposal_bboxes = wml.reshape(proposal_bboxes,shape_box)

    if limits is not None:
        limits = np.array(limits)/np.array(zip(prio_scaling,prio_scaling))
        cy,cx,h,w = tf.unstack(tf.transpose(bboxes_regs,perm=[1,0]),axis=0)
        cy = tf.clip_by_value(cy,clip_value_min=limits[0][0],clip_value_max=limits[0][1])
        cx = tf.clip_by_value(cx,clip_value_min=limits[1][0],clip_value_max=limits[1][1])
        h = tf.clip_by_value(h,clip_value_min=limits[2][0],clip_value_max=limits[2][1])
        w = tf.clip_by_value(w,clip_value_min=limits[3][0],clip_value_max=limits[3][1])
        bboxes_regs = tf.stack([cy,cx,h,w],axis=0)
        bboxes_regs = tf.transpose(bboxes_regs)

    def fn(proposal_bboxes,bboxes_regs,labels,probability):
        boxes = decode_boxes1(proposal_bboxes,bboxes_regs)
        boxes,labels,probability = wlayers.boxes_nms_nr(bboxes=boxes,labels=labels,probs=probability,threshold=nms_threshold,k=candiate_nr,classes_wise=classes_wise_nms)
        return boxes,labels,probability

    boxes,labels,probability = tf.map_fn(lambda x:fn(x[0],x[1],x[2],x[3]),elems=(proposal_bboxes,bboxes_regs,labels,probability),
                                         dtype=(tf.float32,tf.int32,tf.float32),back_prop=False)
    return boxes,labels,probability

'''
this version of get_prediction have no batch dim.

class_prediction:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率
shape为[X,num_classes]

bboxes_regs:模型预测的每个proposal_bboxes/anchro boxes/default boxes到目标真实bbox的回归参数
shape为[X,4](classes_wise=Flase)或者(X,num_classes,4](classes_wise=True)

proposal_bboxes:候选box
shape为[X,4]

limits:[4,2],用于对回归参数的范围进行限制，分别对应于cy,cx,h,w的回归参数，limits的值对应于prio_scaling=[1,1,1,1]是的设置
prio_scaling:在encode_boxes以及decode_boxes是使用的prio_scaling参数

返回:
boxes:[X,4]
labels:[X]
probability:[X]
'''
def __get_predictionv4(class_prediction,
                   bboxes_regs,
                   proposal_bboxes,
                   limits=None,
                   prio_scaling=[0.1,0.1,0.2,0.2],
                   classes_wise=False):
    #删除背景
    class_prediction = class_prediction[:,1:]
    probability,nb_labels = tf.nn.top_k(class_prediction,k=1)
    #背景的类别为0，前面已经删除了背景，需要重新加上
    labels = nb_labels+1
    ndims = class_prediction.get_shape().ndims
    probability = tf.squeeze(probability, axis=ndims - 1)
    labels = tf.squeeze(labels, axis=ndims - 1)

    #按类别在bboxes_regs选择相应类的回归参数
    if classes_wise:
        nb_labels = tf.reshape(nb_labels,[-1])
        bboxes_regs = wml.select_2thdata_by_index(bboxes_regs,nb_labels)

    if limits is not None:
        limits = np.array(limits)/np.array(zip(prio_scaling,prio_scaling))
        cy,cx,h,w = tf.unstack(tf.transpose(bboxes_regs,perm=[1,0]),axis=0)
        cy = tf.clip_by_value(cy,clip_value_min=limits[0][0],clip_value_max=limits[0][1])
        cx = tf.clip_by_value(cx,clip_value_min=limits[1][0],clip_value_max=limits[1][1])
        h = tf.clip_by_value(h,clip_value_min=limits[2][0],clip_value_max=limits[2][1])
        w = tf.clip_by_value(w,clip_value_min=limits[3][0],clip_value_max=limits[3][1])
        bboxes_regs = tf.stack([cy,cx,h,w],axis=0)
        bboxes_regs = tf.transpose(bboxes_regs)

    boxes = decode_boxes1(proposal_bboxes,bboxes_regs)

    return boxes,labels,probability
'''
特点：不删除任何数据，输入与输出数量一致
class_prediction:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率
shape为[batch_size,X,num_classes]

bboxes_regs:模型预测的每个proposal_bboxes/anchro boxes/default boxes到目标真实bbox的回归参数
shape为[batch_size,X,4](classes_wise=Flase)或者(batch_size,X,num_classes,4](classes_wise=True)

proposal_bboxes:候选box
shape为[batch_size,X,4]

threshold:选择class_prediction的阀值

nms_threshold: nms阀值

candiate_nr:选择proposal_bboxes中预测最好的candiate_nr个bboxes

limits:[4,2],用于对回归参数的范围进行限制，分别对应于cy,cx,h,w的回归参数，limits的值对应于prio_scaling=[1,1,1,1]是的设置
prio_scaling:在encode_boxes以及decode_boxes是使用的prio_scaling参数

返回:
boxes:[batch_size,X,4]
labels:[batch_size,X]
probability:[batch_size,X]
'''
def get_predictionv4(class_prediction,
                   bboxes_regs,
                   proposal_bboxes,
                   limits=None,
                   prio_scaling=[0.1,0.1,0.2,0.2],
                   classes_wise=False):
    if proposal_bboxes.get_shape().as_list()[0] == 1:
        '''
        In single stage model, the proposal box are anchor boxes(or default boxes) and for any batch the anchor boxes is the same.
        '''
        proposal_bboxes = tf.squeeze(proposal_bboxes,axis=0)
        boxes,labels,probability= tf.map_fn(lambda x:__get_predictionv4(x[0],x[1],proposal_bboxes,limits,prio_scaling,
                                                                                          classes_wise=classes_wise),
                                                              elems=(class_prediction,bboxes_regs),dtype=(tf.float32,tf.int32,tf.float32)
                                                              )
    else:
        boxes,labels,probability= tf.map_fn(lambda x:__get_predictionv4(x[0],x[1],x[2],limits,prio_scaling,
                                                                                          classes_wise=classes_wise),
                                                              elems=(class_prediction,bboxes_regs,proposal_bboxes),dtype=(tf.float32,tf.int32,tf.float32)
                                                              )
    return boxes,labels,probability

'''
this version of get_prediction have no batch dim.

class_prediction:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率
shape为[X,num_classes]

bboxes_regs:模型预测的每个proposal_bboxes/anchro boxes/default boxes到目标真实bbox的回归参数
shape为[X,4](classes_wise=Flase)或者(X,num_classes,4](classes_wise=True)

proposal_bboxes:候选box
shape为[X,4]

threshold:选择class_prediction的阀值

nms_threshold: nms阀值

candiate_nr:选择proposal_bboxes中预测最好的candiate_nr个bboxes进行筛选

limits:[4,2],用于对回归参数的范围进行限制，分别对应于cy,cx,h,w的回归参数，limits的值对应于prio_scaling=[1,1,1,1]是的设置
prio_scaling:在encode_boxes以及decode_boxes是使用的prio_scaling参数

nms:nms函数,是否使用softnms,使用soft nms与不使用soft nms时, nms_threshold的意义有很大的区别， 不使用soft nms时，nms_threshold表示
IOU小于nms_threshold的两个bbox为不同目标，使用soft nms时，nms_threshold表示得分高于nms_threshold的才是真目标

返回:
boxes:[candiate_nr,4]
labels:[candiate_nr]
probability:[candiate_nr]
len: the available boxes number
'''
def __get_predictionv5(class_prediction,
                   bboxes_regs,
                   proposal_bboxes,
                   limits=None,
                   prio_scaling=[0.1,0.1,0.2,0.2],
                   threshold=0.5,
                   classes_wise=False,
                   candiate_nr = 1500,
                   explicit_remove_background_class=True,
                   max_detection_per_class=100,
                   nms=None):
    #删除背景
    if explicit_remove_background_class:
        class_prediction = class_prediction[:,1:]
    num_classes = class_prediction.get_shape().as_list()[-1]
    r_bboxes = []
    r_labels = []
    r_probs = []
    r_indices = []
    for i in range(num_classes):
        probability= class_prediction[:,i]
        #背景的类别为0，前面已经删除了背景，需要重新加上
        labels = i+1
        res_indices = tf.range(tf.shape(bboxes_regs)[0])
    
        #按类别在bboxes_regs选择相应类的回归参数
        if classes_wise:
            lbboxes_regs = bboxes_regs[:,i,:]
        else:
            lbboxes_regs = tf.identity(bboxes_regs)
        '''
        NMS前数据必须已经排好序
        通过top_k+gather排序
        '''
        probability,indices = tf.nn.top_k(probability,k=tf.shape(probability)[0])
        lbboxes_regs = wml.gather_in_axis(lbboxes_regs,indices,axis=0)
        lproposal_bboxes = wml.gather_in_axis(proposal_bboxes,indices,axis=0)
        res_indices = wml.gather_in_axis(res_indices,indices,axis=0)
    
        pmask = tf.greater(probability,threshold)
        probability = tf.boolean_mask(probability,pmask)
        lproposal_bboxes = tf.boolean_mask(lproposal_bboxes,pmask)
        lboxes_regs = tf.boolean_mask(lbboxes_regs,pmask)
        res_indices = tf.boolean_mask(res_indices,pmask)
        if limits is not None:
            limits = np.array(limits)/np.array(zip(prio_scaling,prio_scaling))
            cy,cx,h,w = tf.unstack(tf.transpose(lboxes_regs,perm=[1,0]),axis=0)
            cy = tf.clip_by_value(cy,clip_value_min=limits[0][0],clip_value_max=limits[0][1])
            cx = tf.clip_by_value(cx,clip_value_min=limits[1][0],clip_value_max=limits[1][1])
            h = tf.clip_by_value(h,clip_value_min=limits[2][0],clip_value_max=limits[2][1])
            w = tf.clip_by_value(w,clip_value_min=limits[3][0],clip_value_max=limits[3][1])
            lboxes_regs = tf.stack([cy,cx,h,w],axis=0)
            lboxes_regs = tf.transpose(lboxes_regs)
    
        boxes = decode_boxes1(lproposal_bboxes,lboxes_regs)
        labels = tf.ones_like(probability)*labels
        boxes,labels,indices = nms(boxes,labels,confidence=probability)
        boxes = boxes[:max_detection_per_class,:]
        labels = labels[:max_detection_per_class]
        indices = indices[:max_detection_per_class]
        probability = tf.gather(probability,indices)
        res_indices = tf.gather(res_indices,indices)
        r_bboxes.append(boxes)
        r_labels.append(labels)
        r_probs.append(probability)
        r_indices.append(res_indices)
        n_v = wmlt.indices_to_dense_vector(res_indices,size=tf.shape(bboxes_regs)[0],indices_value=-1.0,default_value=1.0)
        n_v = tf.reshape(n_v,[-1,1])
        class_prediction = class_prediction*n_v
        ##############################
    
    r_bboxes = tf.concat(r_bboxes,axis=0)
    r_labels = tf.concat(r_labels,axis=0)
    r_probs = tf.concat(r_probs,axis=0)
    r_indices = tf.concat(r_indices,axis=0)
    
    probability,indices = tf.nn.top_k(r_probs,k=tf.minimum(candiate_nr,tf.shape(r_probs)[0]))
    labels = tf.gather(r_labels,indices)
    boxes = tf.gather(r_bboxes,indices)
    res_indices = tf.gather(r_indices,indices)
    len = tf.shape(probability)[0]
    boxes = tf.pad(boxes,paddings=[[0,candiate_nr-len],[0,0]])
    labels = tf.pad(labels,paddings=[[0,candiate_nr-len]])
    probability = tf.pad(probability,paddings=[[0,candiate_nr-len]])
    res_indices = tf.pad(res_indices,paddings=[[0,candiate_nr-len]])
    boxes = tf.reshape(boxes,[candiate_nr,4])
    labels = tf.reshape(labels,[candiate_nr])
    probability = tf.reshape(probability,[candiate_nr])
    res_indices = tf.reshape(res_indices,[candiate_nr])
    return boxes,labels,probability,res_indices,len

'''
class_prediction:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率
shape为[batch_size,X,num_classes]

bboxes_regs:模型预测的每个proposal_bboxes/anchro boxes/default boxes到目标真实bbox的回归参数
shape为[batch_size,X,4](classes_wise=Flase)或者(batch_size,X,num_classes,4](classes_wise=True)

proposal_bboxes:候选box
shape为[batch_size,X,4] (ymin,xmin,ymax,xmax) relative coordinate

threshold:选择class_prediction的阀值

candiate_nr:选择proposal_bboxes中预测最好的candiate_nr个bboxes进行筛选

limits:[4,2],用于对回归参数的范围进行限制，分别对应于cy,cx,h,w的回归参数，limits的值对应于prio_scaling=[1,1,1,1]是的设置
prio_scaling:在encode_boxes以及decode_boxes是使用的prio_scaling参数

返回:
boxes:[batch_size,candiate_nr,4]
labels:[batch_size,candiate_nr]
probability:[batch_size,candiate_nr]
indices:[batch_size,candiate_nr]
len:[batch_size] the available boxes number
'''
def get_predictionv5(class_prediction,
                   bboxes_regs,
                   proposal_bboxes,
                   limits=None,
                   prio_scaling=[0.1,0.1,0.2,0.2],
                   threshold=0.5,
                   candiate_nr = 1500,
                   classes_wise=False,
                   explicit_remove_background_class=True,
                   nms=None):
    if proposal_bboxes.get_shape().as_list()[0] == 1:
        '''
        In single stage model, the proposal box are anchor boxes(or default boxes) and for any batch the anchor boxes is the same.
        '''
        proposal_bboxes = tf.squeeze(proposal_bboxes,axis=0)
        boxes,labels,probability,res_indices,lens = tf.map_fn(lambda x:__get_predictionv5(x[0],x[1],proposal_bboxes,limits,prio_scaling,
                                                                                          threshold,
                                                                                          candiate_nr=candiate_nr,
                                                                                          classes_wise=classes_wise,
                                                                                          explicit_remove_background_class=explicit_remove_background_class,
                                                                                          nms=nms),
                                                              elems=(class_prediction,bboxes_regs),dtype=(tf.float32,tf.int32,tf.float32,tf.int32,tf.int32)
                                                              )
    else:
        boxes,labels,probability,res_indices,lens = tf.map_fn(lambda x:__get_predictionv5(x[0],x[1],x[2],limits,prio_scaling,
                                                                                          threshold,
                                                                                          candiate_nr=candiate_nr,
                                                                                          classes_wise=classes_wise,
                                                                                          explicit_remove_background_class=explicit_remove_background_class,
                                                                                          nms=nms),
                                                              elems=(class_prediction,bboxes_regs,proposal_bboxes),dtype=(tf.float32,tf.int32,tf.float32,tf.int32,tf.int32)
                                                              )
    return boxes,labels,probability,res_indices,lens


'''
与get_predictionv3相同，但NMS作为参数输入同时会返回indices
class_prediction:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率
shape为[batch_size,X,num_classes]

bboxes_regs:模型预测的每个proposal_bboxes/anchro boxes/default boxes到目标真实bbox的回归参数
shape为[batch_size,X,4](classes_wise=Flase)或者(batch_size,X,num_classes,4](classes_wise=True)

proposal_bboxes:候选box
shape为[batch_size,X,4]

threshold:选择class_prediction的阀值

nms_threshold: nms阀值

candiate_nr:选择proposal_bboxes中预测最好的candiate_nr个bboxes

limits:[4,2],用于对回归参数的范围进行限制，分别对应于cy,cx,h,w的回归参数，limits的值对应于prio_scaling=[1,1,1,1]是的设置
prio_scaling:在encode_boxes以及decode_boxes是使用的prio_scaling参数

返回:
boxes:[batch_size,candiate_nr,4]
labels:[batch_size,candiate_nr]
probability:[batch_size,candiate_nr]
'''
def get_predictionv6(class_prediction,
                   bboxes_regs,
                   proposal_bboxes,
                   nms,
                   limits=None,
                   prio_scaling=[0.1,0.1,0.2,0.2],
                   classes_wise=False):
    #删除背景
    class_prediction = class_prediction[:,:,1:]
    probability,nb_labels = tf.nn.top_k(class_prediction,k=1)
    #背景的类别为0，前面已经删除了背景，需要重新加上
    labels = nb_labels+1
    ndims = class_prediction.get_shape().ndims
    probability = tf.squeeze(probability, axis=ndims - 1)
    labels = tf.squeeze(labels, axis=ndims - 1)
    res_indices = tf.reshape(tf.range(tf.shape(labels)[1]),[1,-1])*tf.ones_like(labels,dtype=tf.int32)

    if (isinstance(proposal_bboxes,tf.Tensor) and proposal_bboxes.shape.ndims<3) or (isinstance(proposal_bboxes,np.ndarray) and len(proposal_bboxes.shape)<3):
        proposal_bboxes = tf.expand_dims(proposal_bboxes,axis=0)

    #按类别在bboxes_regs选择相应类的回归参数
    if classes_wise:
        old_bboxes_regs_shape = bboxes_regs.get_shape().as_list()
        nb_labels = tf.reshape(nb_labels,[-1])
        bboxes_regs = tf.reshape(bboxes_regs,[-1,old_bboxes_regs_shape[-2],old_bboxes_regs_shape[-1]])
        bboxes_regs = wml.select_2thdata_by_index(bboxes_regs,nb_labels)
        bboxes_regs = wml.reshape(bboxes_regs,[old_bboxes_regs_shape[0],old_bboxes_regs_shape[1],old_bboxes_regs_shape[3]])
    '''
    NMS前数据必须已经排好序
    通过top_k+gather排序
    '''
    probability,indices = tf.nn.top_k(probability,k=tf.shape(probability)[1])
    shape_2d = labels.get_shape().as_list()
    shape_box = bboxes_regs.get_shape().as_list()
    labels = wml.gather_in_axis(labels,indices,axis=1)
    bboxes_regs = wml.gather_in_axis(bboxes_regs,indices,axis=1)
    proposal_bboxes = wml.gather_in_axis(proposal_bboxes,indices,axis=1)
    res_indices= wml.gather_in_axis(res_indices,indices,axis=1)

    labels = wml.reshape(labels,shape_2d)
    bboxes_regs = wml.reshape(bboxes_regs,shape_box)
    proposal_bboxes = wml.reshape(proposal_bboxes,shape_box)
    res_indices = wml.reshape(res_indices,shape_2d)

    if limits is not None:
        limits = np.array(limits)/np.array(zip(prio_scaling,prio_scaling))
        cy,cx,h,w = tf.unstack(tf.transpose(bboxes_regs,perm=[1,0]),axis=0)
        cy = tf.clip_by_value(cy,clip_value_min=limits[0][0],clip_value_max=limits[0][1])
        cx = tf.clip_by_value(cx,clip_value_min=limits[1][0],clip_value_max=limits[1][1])
        h = tf.clip_by_value(h,clip_value_min=limits[2][0],clip_value_max=limits[2][1])
        w = tf.clip_by_value(w,clip_value_min=limits[3][0],clip_value_max=limits[3][1])
        bboxes_regs = tf.stack([cy,cx,h,w],axis=0)
        bboxes_regs = tf.transpose(bboxes_regs)

    def fn(proposal_bboxes,bboxes_regs,labels,probability):
        boxes = decode_boxes1(proposal_bboxes,bboxes_regs)
        boxes,labels,indices = nms(bboxes=boxes,classes=labels,confidence=probability)
        return boxes,labels,tf.gather(probability,indices),indices

    boxes,labels,probability,indices = tf.map_fn(lambda x:fn(x[0],x[1],x[2],x[3]),elems=(proposal_bboxes,bboxes_regs,labels,probability),
                                         dtype=(tf.float32,tf.int32,tf.float32,tf.int32),back_prop=False)
    res_indices = wmlt.batch_gather(res_indices,indices)
    return boxes,labels,probability,res_indices

'''
this function use nms to remove excess boxes
first boxes is sorted by the probibality of top 1(the background is not counted)
and then use nms to remove boxes to the number of requiremented.
class_prediction:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率
shape为[batch_size,X,num_classes]
bboxes_regs:模型预测的每个proposal_bboxes/anchro boxes/default boxes到目标真实bbox的回归参数
shape为[batch_size,X,4](classes_wise=Flase)或者(batch_size,X,num_classes,4](classes_wise=True)
proposal_bboxes:候选box
shape为[batch_size,X,4]
threshold:选择class_prediction的阀值
nms_threshold: nms阀值
candiate_nr:选择proposal_bboxes中预测最好的candiate_nr个bboxes进行筛选
limits:[4,2],用于对回归参数的范围进行限制，分别对应于cy,cx,h,w的回归参数，limits的值对应于prio_scaling=[1,1,1,1]是的设置
prio_scaling:在encode_boxes以及decode_boxes是使用的prio_scaling参数
返回:
boxes:[Y,4]
labels:[Y]
probability:[Y]
'''
def get_proposal_boxes(class_prediction,
                   bboxes_regs,
                   proposal_bboxes,
                   limits=None,
                   prio_scaling=[0.1,0.1,0.2,0.2],
                   candiate_nr = 1500,
                   candiate_multipler=10,
                   classes_wise=False):
    #删除背景
    class_prediction = class_prediction[:,:,1:]
    probability,nb_labels = tf.nn.top_k(class_prediction,k=1)
    #背景的类别为0，前面已经删除了背景，需要重新加上
    labels = nb_labels+1
    ndims = class_prediction.get_shape().ndims
    probability = tf.squeeze(probability, axis=ndims - 1)
    labels = tf.squeeze(labels, axis=ndims - 1)

    if (isinstance(proposal_bboxes,tf.Tensor) and proposal_bboxes.shape.ndims<3) or (isinstance(proposal_bboxes,np.ndarray) and len(proposal_bboxes.shape)<3):
        proposal_bboxes = tf.expand_dims(proposal_bboxes,axis=0)

    #按类别在bboxes_regs选择相应类的回归参数
    if classes_wise:
        old_bboxes_regs_shape = bboxes_regs.get_shape()
        nb_labels = tf.reshape(nb_labels,[-1])
        bboxes_regs = tf.reshape(bboxes_regs,[-1,old_bboxes_regs_shape[-2],old_bboxes_regs_shape[-1]])
        bboxes_regs = wml.select_2thdata_by_index(bboxes_regs,nb_labels)
        bboxes_regs = tf.reshape(bboxes_regs,[old_bboxes_regs_shape[0],old_bboxes_regs_shape[1],old_bboxes_regs_shape[3]])
    '''
    NMS前数据必须已经排好序
    通过top_k+gather排序
    '''
    probability,indices = tf.nn.top_k(probability,k=tf.minimum(tf.shape(probability)[1],candiate_nr*candiate_multipler))
    labels = wml.gather_in_axis(labels,indices,axis=1)
    bboxes_regs = wml.gather_in_axis(bboxes_regs,indices,axis=1)
    proposal_bboxes = wml.gather_in_axis(proposal_bboxes,indices,axis=1)

    if limits is not None:
        limits = np.array(limits)/np.array(zip(prio_scaling,prio_scaling))
        cy,cx,h,w = tf.unstack(tf.transpose(bboxes_regs,perm=[1,0]),axis=0)
        cy = tf.clip_by_value(cy,clip_value_min=limits[0][0],clip_value_max=limits[0][1])
        cx = tf.clip_by_value(cx,clip_value_min=limits[1][0],clip_value_max=limits[1][1])
        h = tf.clip_by_value(h,clip_value_min=limits[2][0],clip_value_max=limits[2][1])
        w = tf.clip_by_value(w,clip_value_min=limits[3][0],clip_value_max=limits[3][1])
        bboxes_regs = tf.stack([cy,cx,h,w],axis=0)
        bboxes_regs = tf.transpose(bboxes_regs)

    def fn(proposal_bboxes,bboxes_regs,labels,probability):
        boxes = decode_boxes1(proposal_bboxes,bboxes_regs)
        boxes,labels,indices = boxes_nms_nr(boxes,labels,k=candiate_nr,max_loop=5)
        probability = tf.gather(probability,indices)
        return boxes,labels,probability

    boxes,labels,probability = tf.map_fn(lambda x:fn(x[0],x[1],x[2],x[3]),elems=(proposal_bboxes,bboxes_regs,labels,probability),
                                         dtype=(tf.float32,tf.int32,tf.float32),back_prop=False)
    return boxes,labels,probability

'''
this function get exactly candiate_nr boxes by the probability
class_prediction:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率
shape为[batch_size,X,num_classes]
bboxes_regs:模型预测的每个proposal_bboxes/anchro boxes/default boxes到目标真实bbox的回归参数
shape为[batch_size,X,4](classes_wise=Flase)或者(batch_size,X,num_classes,4](classes_wise=True)
proposal_bboxes:候选box
shape为[batch_size,X,4]
threshold:选择class_prediction的阀值
nms_threshold: nms阀值
candiate_nr:选择proposal_bboxes中预测最好的candiate_nr个bboxes进行筛选
limits:[4,2],用于对回归参数的范围进行限制，分别对应于cy,cx,h,w的回归参数，limits的值对应于prio_scaling=[1,1,1,1]是的设置
prio_scaling:在encode_boxes以及decode_boxes是使用的prio_scaling参数
返回:
boxes:[Y,4]
labels:[Y]
probability:[Y]
'''
def get_proposal_boxesv2(class_prediction,
                   bboxes_regs,
                   proposal_bboxes,
                   limits=None,
                   prio_scaling=[0.1,0.1,0.2,0.2],
                   candiate_nr = 1500,
                   classes_wise=False):

    if not bboxes_regs.get_shape().is_fully_defined():
        shape = bboxes_regs.get_shape().as_list()
        d_shape = tf.shape(bboxes_regs)
        if shape[0] is None and proposal_bboxes.get_shape().is_fully_defined():
            if proposal_bboxes.get_shape().ndims == 2:
                proposal_bboxes = tf.expand_dims(proposal_bboxes,axis=0)
            proposal_bboxes = proposal_bboxes*tf.ones(shape=[d_shape[0],d_shape[1],4])

    #删除背景
    class_prediction = class_prediction[:,:,1:]
    probability,nb_labels = tf.nn.top_k(class_prediction,k=1)
    #背景的类别为0，前面已经删除了背景，需要重新加上
    labels = nb_labels+1
    ndims = class_prediction.get_shape().ndims
    probability = tf.squeeze(probability, axis=ndims - 1)
    labels = tf.squeeze(labels, axis=ndims - 1)

    if (isinstance(proposal_bboxes,tf.Tensor) and proposal_bboxes.shape.ndims<3) or (isinstance(proposal_bboxes,np.ndarray) and len(proposal_bboxes.shape)<3):
        proposal_bboxes = tf.expand_dims(proposal_bboxes,axis=0)

    #按类别在bboxes_regs选择相应类的回归参数
    if classes_wise:
        old_bboxes_regs_shape = bboxes_regs.get_shape().as_list()
        nb_labels = tf.reshape(nb_labels,[-1])
        bboxes_regs = tf.reshape(bboxes_regs,[-1,old_bboxes_regs_shape[-2],old_bboxes_regs_shape[-1]])
        bboxes_regs = wml.select_2thdata_by_index(bboxes_regs,nb_labels)
        bboxes_regs = wmlt.reshape(bboxes_regs,[old_bboxes_regs_shape[0],old_bboxes_regs_shape[1],old_bboxes_regs_shape[3]])

    '''
    通过top_k+gather排序
    '''
    probability,indices = tf.nn.top_k(probability,k=tf.minimum(candiate_nr,tf.shape(probability)[1]))
    shape_2d = labels.get_shape().as_list()
    shape_box = bboxes_regs.get_shape().as_list()
    labels = wml.batch_gather(labels,indices)
    bboxes_regs = wml.batch_gather(bboxes_regs,indices)
    proposal_bboxes = wml.batch_gather(proposal_bboxes,indices)

    if limits is not None:
        limits = np.array(limits)/np.array(zip(prio_scaling,prio_scaling))
        cy,cx,h,w = tf.unstack(tf.transpose(bboxes_regs,perm=[1,0]),axis=0)
        cy = tf.clip_by_value(cy,clip_value_min=limits[0][0],clip_value_max=limits[0][1])
        cx = tf.clip_by_value(cx,clip_value_min=limits[1][0],clip_value_max=limits[1][1])
        h = tf.clip_by_value(h,clip_value_min=limits[2][0],clip_value_max=limits[2][1])
        w = tf.clip_by_value(w,clip_value_min=limits[3][0],clip_value_max=limits[3][1])
        bboxes_regs = tf.stack([cy,cx,h,w],axis=0)
        bboxes_regs = tf.transpose(bboxes_regs)

    def fn(proposal_bboxes,bboxes_regs,labels,probability):
        boxes = decode_boxes1(proposal_bboxes,bboxes_regs)
        #boxes = boxes[:candiate_nr]
        #labels = labels[:candiate_nr]
        #probability = probability[:candiate_nr]
        boxes = tf.reshape(boxes,[candiate_nr,4])
        labels = tf.reshape(labels,[candiate_nr])
        probability = tf.reshape(probability,[candiate_nr])
        return boxes,labels,probability

    boxes,labels,probability = tf.map_fn(lambda x:fn(x[0],x[1],x[2],x[3]),elems=(proposal_bboxes,bboxes_regs,labels,probability),
                                         dtype=(tf.float32,tf.int32,tf.float32),back_prop=False)
    return boxes,labels,probability


'''
this function get exactly candiate_nr boxes by the heristic method
firt remove boxes by nums, and then get the top candiate_nr boxes, if there not enough boxes after nms,
the boxes in the front will be add to result.
class_prediction:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率
shape为[batch_size,X,num_classes]
bboxes_regs:模型预测的每个proposal_bboxes/anchro boxes/default boxes到目标真实bbox的回归参数
shape为[batch_size,X,4](classes_wise=Flase)或者(batch_size,X,num_classes,4](classes_wise=True)
proposal_bboxes:候选box
shape为[batch_size,X,4]
threshold:选择class_prediction的阀值
nms_threshold: nms阀值
candiate_nr:选择proposal_bboxes中预测最好的candiate_nr个bboxes进行筛选
limits:[4,2],用于对回归参数的范围进行限制，分别对应于cy,cx,h,w的回归参数，limits的值对应于prio_scaling=[1,1,1,1]是的设置
prio_scaling:在encode_boxes以及decode_boxes是使用的prio_scaling参数
返回:
boxes:[Y,4]
labels:[Y]
probability:[Y]
'''
def get_proposal_boxesv3(class_prediction,
                   bboxes_regs,
                   proposal_bboxes,
                   limits=None,
                   prio_scaling=[0.1,0.1,0.2,0.2],
                   candiate_nr = 1500,
                   nms_threshold=0.8,
                   classes_wise=False):

    if not bboxes_regs.get_shape().is_fully_defined():
        shape = bboxes_regs.get_shape().as_list()
        d_shape = tf.shape(bboxes_regs)
        if shape[0] is None and proposal_bboxes.get_shape().is_fully_defined():
            if proposal_bboxes.get_shape().ndims == 2:
                proposal_bboxes = tf.expand_dims(proposal_bboxes,axis=0)
            proposal_bboxes = proposal_bboxes*tf.ones(shape=[d_shape[0],d_shape[1],4])

    #删除背景
    class_prediction = class_prediction[:,:,1:]
    probability,nb_labels = tf.nn.top_k(class_prediction,k=1)
    #背景的类别为0，前面已经删除了背景，需要重新加上
    labels = nb_labels+1
    ndims = class_prediction.get_shape().ndims
    probability = tf.squeeze(probability, axis=ndims - 1)
    labels = tf.squeeze(labels, axis=ndims - 1)

    if (isinstance(proposal_bboxes,tf.Tensor) and proposal_bboxes.shape.ndims<3) or (isinstance(proposal_bboxes,np.ndarray) and len(proposal_bboxes.shape)<3):
        proposal_bboxes = tf.expand_dims(proposal_bboxes,axis=0)

    #按类别在bboxes_regs选择相应类的回归参数
    if classes_wise:
        old_bboxes_regs_shape = bboxes_regs.get_shape().as_list()
        nb_labels = tf.reshape(nb_labels,[-1])
        bboxes_regs = tf.reshape(bboxes_regs,[-1,old_bboxes_regs_shape[-2],old_bboxes_regs_shape[-1]])
        bboxes_regs = wml.select_2thdata_by_index(bboxes_regs,nb_labels)
        bboxes_regs = wmlt.reshape(bboxes_regs,[old_bboxes_regs_shape[0],old_bboxes_regs_shape[1],old_bboxes_regs_shape[3]])

    '''
    通过top_k+gather排序
    '''
    probability,indices = tf.nn.top_k(probability,k=tf.minimum(candiate_nr*3,tf.shape(probability)[1]))
    labels = wml.batch_gather(labels,indices)
    bboxes_regs = wml.batch_gather(bboxes_regs,indices)
    proposal_bboxes = wml.batch_gather(proposal_bboxes,indices)

    if limits is not None:
        limits = np.array(limits)/np.array(zip(prio_scaling,prio_scaling))
        cy,cx,h,w = tf.unstack(tf.transpose(bboxes_regs,perm=[1,0]),axis=0)
        cy = tf.clip_by_value(cy,clip_value_min=limits[0][0],clip_value_max=limits[0][1])
        cx = tf.clip_by_value(cx,clip_value_min=limits[1][0],clip_value_max=limits[1][1])
        h = tf.clip_by_value(h,clip_value_min=limits[2][0],clip_value_max=limits[2][1])
        w = tf.clip_by_value(w,clip_value_min=limits[3][0],clip_value_max=limits[3][1])
        bboxes_regs = tf.stack([cy,cx,h,w],axis=0)
        bboxes_regs = tf.transpose(bboxes_regs)

    def fn(proposal_bboxes,bboxes_regs,labels,probability):
        boxes = decode_boxes1(proposal_bboxes,bboxes_regs)
        boxes,labels,indices = boxes_nms_nr2(boxes,labels,k=candiate_nr,threshold=nms_threshold)
        #boxes,labels,probability = wlayers.boxes_nms_nr(bboxes=boxes,labels=labels,probs=probability,threshold=nms_threshold,k=candiate_nr)
        #probability = tf.gather(probability,indices)
        return boxes,labels,probability

    boxes,labels,probability = tf.map_fn(lambda x:fn(x[0],x[1],x[2],x[3]),elems=(proposal_bboxes,bboxes_regs,labels,probability),
                                         dtype=(tf.float32,tf.int32,tf.float32),back_prop=False)
    return tf.stop_gradient(boxes),tf.stop_gradient(labels),tf.stop_gradient(probability)

def get_prediction_no_background(class_prediction,
                   bboxes_regs,
                   proposal_bboxes,
                   threshold=0.5,
                   nms_threshold=0.1,
                   candiate_nr = 1500,
                   classes_wise=False):
    probability,labels = tf.nn.top_k(class_prediction,k=1)
    ndims = class_prediction.get_shape().ndims
    probability = tf.squeeze(probability, axis=ndims - 1)
    labels = tf.squeeze(labels, axis=ndims - 1)
    if (isinstance(proposal_bboxes,tf.Tensor) and proposal_bboxes.shape.ndims<3) or (isinstance(proposal_bboxes,np.ndarray) and len(proposal_bboxes.shape)<3):
        proposal_bboxes = tf.expand_dims(proposal_bboxes,axis=0)
    batch_size = class_prediction.get_shape().as_list()[0]

    if classes_wise:
        old_bboxes_regs_shape = bboxes_regs.get_shape()
        old_labels_shape = labels.get_shape()
        labels = tf.reshape(labels,[-1])
        bboxes_regs = tf.reshape(bboxes_regs,[-1,old_bboxes_regs_shape[-2],old_bboxes_regs_shape[-1]])
        bboxes_regs = wml.select_2thdata_by_index(bboxes_regs,labels)
        bboxes_regs = tf.reshape(bboxes_regs,[old_bboxes_regs_shape[0],-1,old_bboxes_regs_shape[3]])
        labels = tf.reshape(labels,[old_labels_shape[0],-1])
    '''
    NMS前数据必须已经排好序
    通过top_k+gather排序
    '''
    probability,indices = tf.nn.top_k(probability,k=tf.minimum(candiate_nr,tf.shape(probability)[1]))
    labels = wml.gather_in_axis(labels,indices,axis=1)
    bboxes_regs = wml.gather_in_axis(bboxes_regs,indices,axis=1)
    proposal_bboxes = wml.gather_in_axis(proposal_bboxes,indices,axis=1)

    pmask = tf.greater(probability,threshold)
    probability = tf.boolean_mask(probability,pmask)
    labels = tf.boolean_mask(labels,pmask)
    proposal_bboxes = tf.boolean_mask(proposal_bboxes,pmask)
    boxes_regs = tf.boolean_mask(bboxes_regs,pmask)
    boxes = decode_boxes1(proposal_bboxes,boxes_regs)
    boxes,labels,probability = boxesNMS(boxes,labels,probability,threshold=nms_threshold)
    return boxes,labels,probability

'''
boxes:[X,4] ymin,xmin,ymax,xmax
labels:[X]
probability:[X]
'''
def get_prediction_boxes(boxes,labels,probability,
                   threshold=0.5,
                   nms_threshold=0.1,
                   candiate_nr = 1500,
                   classes_wise=True):


    '''
    NMS前数据必须已经排好序
    通过top_k+gather排序
    '''
    probability,indices = tf.nn.top_k(probability,k=tf.minimum(candiate_nr,tf.shape(probability)[0]))
    labels = wml.gather_in_axis(labels,indices,axis=0)
    boxes = wml.gather_in_axis(boxes,indices,axis=0)

    pmask = tf.greater(probability,threshold)
    probability = tf.boolean_mask(probability,pmask)
    labels = tf.boolean_mask(labels,pmask)
    boxes = tf.boolean_mask(boxes,pmask)
    boxes,labels,indices = boxes_nms(boxes,labels,threshold=nms_threshold,classes_wise=classes_wise)
    probability = tf.gather(probability,indices)
    return boxes,labels,probability

'''
image: A 3-D tensor of shape [height, width, channels].
bboxes:[X,4] X个bboxes,使用相对坐标，[ymin,xmin,ymax,xmax]
'''
def flip_left_right(image,bboxes):
    if isinstance(image,list):
        image = [tf.cond(tf.reduce_min(tf.shape(img))>0,lambda:tf.image.flip_left_right(img),lambda:img) for img in image]
    else:
        image = tf.image.flip_left_right(image)
    return image,bboxes_flip_left_right(bboxes)

def bboxes_flip_left_right(bboxes):
    bboxes = tf.transpose(bboxes)
    ymin,xmin,ymax,xmax = tf.unstack(bboxes,axis=0)
    nxmax = 1.0-xmin
    nxmin = 1.0-xmax
    bboxes = tf.stack([ymin,nxmin,ymax,nxmax],axis=0)
    bboxes = tf.transpose(bboxes)
    return bboxes

def random_flip_left_right(image,bboxes):
    return tf.cond(tf.greater(tf.random_uniform(shape=[]), 0.5),
            lambda: (image, bboxes),
            lambda: flip_left_right(image,bboxes))
'''
image: A 3-D tensor of shape [height, width, channels].
bboxes:[X,4] X个bboxes,使用相对坐标，[ymin,xmin,ymax,xmax]
'''
def flip_up_down(image,bboxes):
    if isinstance(image,list):
        image = [tf.cond(tf.reduce_min(tf.shape(img))>0,lambda:tf.image.flip_up_down(img),lambda:img) for img in image]
    else:
        image = tf.image.flip_up_down(image)
    return image,bboxes_flip_up_down(bboxes)

def bboxes_flip_up_down(bboxes):
    bboxes = tf.transpose(bboxes)
    ymin,xmin,ymax,xmax = tf.unstack(bboxes,axis=0)
    nymax = 1.0-ymin
    nymin = 1.0- ymax
    bboxes = tf.stack([nymin,xmin,nymax,xmax],axis=0)
    bboxes = tf.transpose(bboxes)
    return bboxes

def random_flip_up_down(image,bboxes):
    return tf.cond(tf.greater(tf.random_uniform(shape=[]), 0.5),
                   lambda: (image, bboxes),
                   lambda: flip_up_down(image,bboxes))

def rot90(image,bboxes,clockwise=True):
    if clockwise:
        k = 1
    else:
        k = 3
    if isinstance(image,list):
        image = [tf.cond(tf.reduce_min(tf.shape(img))>0,lambda:tf.image.rot90(img,k),lambda:img) for img in image]
    else:
        image = tf.image.rot90(image,k)
    return image,bboxes_rot90(bboxes,clockwise)

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

def bboxes_rot90_ktimes(bboxes,clockwise=True,k=1):
    i = tf.constant(0)
    c = lambda i,box:tf.less(i,k)
    b = lambda i,box:(i+1,bboxes_rot90(box,clockwise))
    _,box = tf.while_loop(c,b,loop_vars=[i,bboxes])
    return box

def random_rot90(image,bboxes,clockwise=True):
    return tf.cond(tf.greater(tf.random_uniform(shape=[]), 0.5),
                   lambda: (image, bboxes),
                   lambda: rot90(image,bboxes,clockwise))

'''
dst为原图中box指定区域的mask
'''
def merge_mask_in_box(src,dst,box):
    assert(src.shape[2]==dst.shape[2])
    src_h,src_w,c = src.shape
    box_w = box[3]-box[1]
    box_h = box[2]-box[0]

    if box_w<=0 or box_h<=0:
        return src

    d2s_w = src_w*box_w
    d2s_h = src_h*box_h
    x_beg = box[1]*src_h
    y_beg = box[0]*src_w
    x_beg = np.clip(x_beg,0,src_w-1)
    y_beg = np.clip(y_beg,0,src_h-1)
    d2s_w = np.clip(d2s_w,1,src_w-x_beg)
    d2s_h = np.clip(d2s_h,1,src_h-y_beg)
    dst = img_utils.resize_img(dst,[d2s_h,d2s_w])

    src[y_beg:y_beg+d2s_h,x_beg:x_beg+d2s_w] = dst

    return src
def distored_boxes(boxes,scale=[],xoffset=[],yoffset=[]):
    if boxes.get_shape().ndims == 3:
        return tf.map_fn(lambda x:wop.distored_boxes(x,scale=scale, xoffset=xoffset, yoffset=yoffset),elems=boxes)
    else:
        return wop.distored_boxes(boxes,scale=scale, xoffset=xoffset, yoffset=yoffset)

'''
boxes:[N,4]
'''
def random_select_boxes(boxes,size,len=None):
    with tf.name_scope("random_select_boxes"):
        if len is None:
            data_nr = tf.shape(boxes)[0]
        else:
            data_nr = len
        indexs = tf.range(data_nr)
        indexs = wop.wpad(indexs, [0, size - data_nr])
        indexs = tf.random_shuffle(indexs)
        indexs = tf.random_crop(indexs, [size])
        boxes = tf.gather(boxes, indexs)
        return boxes, indexs
'''
boxes:[batch_size,N,4]
lens:[batch_size]
'''
def random_select_boxes_v2(boxes,lens,size):
    with tf.name_scope("random_select_boxes"):
        boxes,indexs = tf.map_fn(lambda x:random_select_boxes(x[0],size,x[1]),elems=(boxes,lens),dtype=(tf.float32,tf.int32))
        batch_size = boxes.get_shape().as_list()[0]
        boxes = wml.reshape(boxes,[batch_size,size,4])
        indexs = wml.reshape(indexs,[batch_size,size])
        return boxes,indexs
'''
Select boxes by probability threshold
bboxes:[N,4]
labels:[N]
probs:[N]
threshold: ()
'''
def select_boxes_by_probs(bboxes,labels,probs,threshold=0.5):
    mask = probs>threshold
    bboxes = tf.boolean_mask(bboxes,mask)
    labels = tf.boolean_mask(labels,mask)
    probs = tf.boolean_mask(probs,mask)
    return bboxes,labels,probs

def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered=0.3,
                                aspect_ratio_range=(0.9, 1.1),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                filter_threshold=0.3,
                                scope=None):
    '''
    data argument for object detection
    :param image: [height,width,channels], input image
    :param labels: [num_boxes]
    :param bboxes: [num_boxes,4] (ymin,xmin,ymax,xmax), relative coordinates
    :param min_object_covered: the croped object bbox and the orignal object bbox's IOU must greater than
    min_object_covered to be keep in the result, else thre object will be removed.
    :param aspect_ratio_range: random crop aspect ratio range
    :param area_range:
    :param max_attempts:
    :param scope:
    :return:
    croped_image:[h,w,C]
    labels:[n]
    bboxes[n,4]
    mask:[num_boxes]
    bbox_begin:[3]
    bbox_size:[3]
    '''
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes]):
        '''
        在满足一定要求的前提下对图片进行随机的裁剪
        '''
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=tf.expand_dims(bboxes, 0),
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True,
                seed=int(time.time()),
                seed2=int(10*time.time()))
        '''
        distort_bbox的shape一定为[1, 1, 4],表示需要裁剪的裁剪框，与bbox_begin,bbox_size表达的内容重复
        '''
        distort_bbox = distort_bbox[0, 0]

        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        cropped_image.set_shape([None, None, 3])

        '''
        将在原图中的bboxes转换为在distort_bbox定义的子图中的bboxes
        保留了distort_bbox界外的部分
        '''
        bboxes = wml_bboxes.bboxes_resize(distort_bbox, bboxes)
        '''
        仅保留交叉面积大于threshold的bbox
        '''
        labels, bboxes,mask = wml_bboxes.bboxes_filter_overlap(labels, bboxes,
                                                   threshold=filter_threshold,
                                                   assign_negative=False)
        return cropped_image, labels, bboxes,bbox_begin,bbox_size,mask

'''
mask:[N,H,W]
'''
def distorted_bounding_box_and_mask_crop(image,
                                mask,
                                labels,
                                bboxes,
                                min_object_covered=0.3,
                                aspect_ratio_range=(0.9, 1.1),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                filter_threshold=0.3,
                                scope=None):
    dst_image, labels, bboxes, bbox_begin,bbox_size,bboxes_mask= \
            distorted_bounding_box_crop(image, labels, bboxes,
                                        area_range=area_range,
                                        min_object_covered=min_object_covered,
                                        aspect_ratio_range=aspect_ratio_range,
                                        max_attempts=max_attempts,
                                        filter_threshold=filter_threshold,
                                        scope=scope)
    mask = tf.boolean_mask(mask,bboxes_mask)
    mask = tf.transpose(mask,perm=[1,2,0])
    mask = tf.slice(mask,bbox_begin,bbox_size)
    dst_mask = tf.transpose(mask,perm=[2,0,1])
    return dst_image,dst_mask,labels, bboxes, bbox_begin,bbox_size,bboxes_mask


