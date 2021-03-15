#coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import random
import os
import object_detection.npod_toolkit as npod
import math
from semantic.visualization_utils import draw_bounding_boxes_on_image_tensors
import object_detection.visualization as odv
import wml_utils
import logging
import shutil
from object_detection.metrics import coco_evaluation
from object_detection.metrics import standard_fields
import wml_utils as wmlu
import img_utils as wmli
import copy
import iotoolkit.pascal_voc_toolkit as voc
read_voc_xml = voc.read_voc_xml
write_voc_xml = voc.write_voc_xml
getVOCFiles = voc.getVOCFiles
getVOCFilesV2 = voc.getVOCFilesV2
writeVOCXml = voc.writeVOCXml
writeVOCXmlV2 = voc.writeVOCXmlV2
writeVOCXmlByImg = voc.writeVOCXmlByImg
filterVOCFilesByName = voc.filterVOCFilesByName
removeUnmatchVOCFiles = voc.removeUnmatchVOCFiles

'''
image:[h,w,c], value range(0,255) if scale is True else (0,1)
boxes:[X,4],relative coordinate, (ymin,xmin,ymax,xmax)
'''
def tf_draw_image_with_box(image, bboxes, name='summary_image_with_box',scale=True):
    if scale:
        if image.dtype != tf.float32:
            image = tf.cast(image,tf.float32)
        image = image/255.
    if image.get_shape().ndims == 3:
        image = tf.expand_dims(image,axis=0)
    if image.dtype is not tf.float32:
        image = tf.cast(image,tf.float32)
    if bboxes.get_shape().ndims == 2:
        bboxes = tf.expand_dims(bboxes, 0)
    image_with_box = tf.image.draw_bounding_boxes(image, bboxes)
    return image_with_box
'''
image:[h,w,c], value range(0,255) if scale is True else (0,1)
boxes:[X,4],relative coordinate, (ymin,xmin,ymax,xmax)
'''
def tf_summary_image_with_box(image, bboxes, name='summary_image_with_box',scale=True):
    with tf.name_scope(name):
        image_with_box = tf_draw_image_with_box(image,bboxes,name,scale)
        tf.summary.image(name, image_with_box)

def tf_summary_image_with_boxv2(image,
                                bboxes,
                                classes,
                                scores=None,
                                category_index=None,
                                max_boxes_to_draw=64,
                                name='summary_image_with_box',scale=True):
    with tf.name_scope(name):
        if scale:
            if (image.dtype==tf.float32) or (image.dtype==tf.float16) or (image.dtype==tf.float64):
                #floatpoint data value range is [-1,1]
                image = (image+1.0)*127.5
                image = tf.clip_by_value(image,clip_value_min=0.,clip_value_max=255.)
                image = tf.cast(image,dtype=tf.uint8)
        if image.get_shape().ndims == 3:
            image = tf.expand_dims(image,axis=0)
        if image.dtype is not tf.uint8:
            image = tf.cast(image,tf.uint8)
        if bboxes.get_shape().ndims == 2:
            bboxes = tf.expand_dims(bboxes, 0)
        if scores is None:
            scores = tf.ones(shape=tf.shape(bboxes)[:2],dtype=tf.float32)
        elif scores.get_shape().ndims ==1:
            scores = tf.expand_dims(scores,axis=0)
        if category_index is None:
            category_index = {}
            for i in range(100):
                category_index[i] = {"name":f"{i}"}
        if classes.get_shape().ndims == 1:
            classes = tf.expand_dims(classes,axis=0)
        image_with_box = draw_bounding_boxes_on_image_tensors(image,bboxes,classes,scores,
                                                              category_index=category_index,
                                                              max_boxes_to_draw=max_boxes_to_draw)
        tf.summary.image(name, image_with_box)

def tf_summary_image_with_boxv3(image,
                                bboxes,
                                classes,
                                scores=None,
                                get_color_func=None,
                                name='summary_image_with_box',scale=True):
    with tf.name_scope(name):
        if scale:
            if (image.dtype==tf.float32) or (image.dtype==tf.float16) or (image.dtype==tf.float64):
                #floatpoint data value range is [-1,1]
                image = (image+1.0)*127.5
                image = tf.clip_by_value(image,clip_value_min=0.,clip_value_max=255.)
                image = tf.cast(image,dtype=tf.uint8)
        if image.dtype is tf.uint8:
            image = tf.cast(image,tf.uint8)
        if scores is None:
            scores = tf.ones(shape=tf.shape(bboxes)[:2],dtype=tf.float32)
        def draw_func(img,label,score,boxes):
            return odv.bboxes_draw_on_imgv2(img,label,score,boxes,get_color_func,show_text=False)
        image_with_box = tf.py_func(draw_func,(image,classes,scores,bboxes),image.dtype)
        image_with_box = tf.expand_dims(image_with_box,axis=0)
        tf.summary.image(name, image_with_box)

def tf_summary_image_with_boxs_and_lens(image, bboxes, lens,name='summary_image_with_box',scale=True):
    with tf.name_scope(name):
        if scale:
            if image.dtype != tf.float32:
                image = tf.cast(image,tf.float32)
            image = image/255.
        if image.get_shape().ndims == 3:
            image = tf.expand_dims(image,axis=0)
        if image.dtype is not tf.float32:
            image = tf.cast(image,tf.float32)
        if bboxes.get_shape().ndims == 2:
            bboxes = tf.expand_dims(bboxes, 0)
        def fn(img,boxes,len):
            img = tf.expand_dims(img,axis=0)
            boxes = tf.expand_dims(boxes[:len,:],axis=0)
            img = tf.image.draw_bounding_boxes(img, boxes)
            return tf.squeeze(img,axis=0)
        image_with_box = tf.map_fn(lambda x:fn(x[0],x[1],x[2]),elems=(image,bboxes,lens),dtype=tf.float32)
        tf.summary.image(name, image_with_box)

def tf_summary_image_with_boxs_and_masks(image, bboxes, masks,name='summary_image_with_box',scale=True):
    with tf.name_scope(name):
        if scale:
            if image.dtype != tf.float32:
                image = tf.cast(image,tf.float32)
            image = image/255.
        if image.get_shape().ndims == 3:
            image = tf.expand_dims(image,axis=0)
        if image.dtype is not tf.float32:
            image = tf.cast(image,tf.float32)
        if bboxes.get_shape().ndims == 2:
            bboxes = tf.expand_dims(bboxes, 0)
        def fn(img,boxes,mask):
            img = tf.expand_dims(img,axis=0)
            boxes = tf.expand_dims(tf.boolean_mask(boxes,mask),axis=0)
            img = tf.image.draw_bounding_boxes(img, boxes)
            return tf.squeeze(img,axis=0)
        image_with_box = tf.map_fn(lambda x:fn(x[0],x[1],x[2]),elems=(image,bboxes,masks),dtype=tf.float32)
        tf.summary.image(name, image_with_box)

def tf_summary_image_grayscale_with_box(image, bboxes, name='image',scale=True):
    with tf.name_scope(name):
        if scale:
            image = tf.expand_dims(image, 0)/255.
        image = tf.image.grayscale_to_rgb(image)
        if image.get_shape().ndims == 3:
            image = tf.expand_dims(image, 0)
            bboxes = tf.expand_dims(bboxes, 0)
        image_with_box = image
        for boxes in bboxes:
            shape = boxes.get_shape().as_list()
            boxes = tf.reshape(boxes,[shape[0],-1,shape[-1]])
            image_with_box = tf.image.draw_bounding_boxes(image_with_box, boxes)
        tf.summary.image(name, image)

def get_shape(x, rank=None):
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape()
        if rank is None:
            static_shape = static_shape.as_list()
            rank = len(static_shape)
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def __safe_persent(v0,v1):
    if v1==0:
        return 100.
    else:
        return v0*100./v1

def getF1(gtboxes,gtlabels,boxes,labels,threshold=0.5):
    gt_shape = gtboxes.shape
    #indict if there have some box match with this ground-truth box
    gt_mask = np.zeros([gt_shape[0]],dtype=np.int32)
    boxes_shape = boxes.shape
    #indict if there have some ground-truth box match with this box
    boxes_mask = np.zeros(boxes_shape[0],dtype=np.int32)
    gt_size = gtlabels.shape[0]
    boxes_size = labels.shape[0]
    for i in range(gt_size):
        max_index = -1
        max_jaccard = 0.0
        #iterator on all boxes to find one have the most maximum jacard value with current ground-truth box
        for j in range(boxes_size):
            if gtlabels[i] != labels[j] or boxes_mask[j] != 0:
                continue
            jaccard = npod.box_jaccard(gtboxes[i],boxes[j])
            if jaccard>threshold and jaccard > max_jaccard:
                max_jaccard = jaccard
                max_index = j

        if max_index < 0:
            continue

        gt_mask[i] = 1
        boxes_mask[max_index] = 1

    correct_num = np.sum(gt_mask)
    f1 = __safe_persent(2*correct_num,correct_num+gt_shape[0])

    return f1

'''
gtboxes:[X,4](ymin,xmin,ymax,xmax) relative coordinates, ground truth boxes
gtlabels:[X] the labels for ground truth boxes
boxes:[Y,4](ymin,xmin,ymax,xmax) relative coordinates,predicted boxes
labels:[Y], the labels for predicted boxes
probability:[Y], the probability for boxes, if probability is none, assum the boxes's probability is ascending order
'''
def getmAP(gtboxes,gtlabels,boxes,labels,probability=None,threshold=0.5):

    if not isinstance(gtboxes,np.ndarray):
        gtboxes = np.array(gtboxes)
    if not isinstance(gtlabels,np.ndarray):
        gtlabels = np.array(gtlabels)
    if not isinstance(boxes,np.ndarray):
        boxes = np.array(boxes)
    if not isinstance(labels,np.ndarray):
        labels = np.array(labels)
    gtboxes = copy.deepcopy(np.array(gtboxes))
    gtlabels = copy.deepcopy(np.array(gtlabels))
    boxes = copy.deepcopy(boxes)
    labels = copy.deepcopy(labels)
    if probability is not None:
        probability = copy.deepcopy(probability)
        index = np.argsort(probability)
        boxes = boxes[index]
        labels = labels[index]

    max_nr = 20
    data_nr = boxes.shape[0]

    if data_nr==0:
        return 0.0

    if data_nr>max_nr:
        beg_index = range(0,data_nr,data_nr//max_nr)
    else:
        beg_index = range(0,data_nr)

    res = []

    for v in beg_index:
        p,r = getPrecision(gtboxes,gtlabels,boxes[v:],labels[v:],threshold)
        res.append([p,r])

    res.sort(key=lambda x:x[1])
    #print("mAP:res: {}".format(res))

    min_r = res[0][1]
    max_r = res[-1][1]
    logging.debug("mAP: max r {}, min r {}".format(max_r,min_r))
    if max_r-min_r<1.0:
        p,r = getPrecision(gtboxes,gtlabels,boxes,labels,threshold)
        res = [[p,r],[p,r]]

    if min_r > 1e-2:
        res = np.concatenate([np.array([[res[0][0],0.]]),res],axis=0)
    if max_r <100.0-1e-2:
        if max_r+10.<100.0:
            res = np.concatenate([res,np.array([[0.,max_r+10.],[0.,100.]])])
        else:
            res = np.concatenate([res,np.array([[0.,max_r+10.]])])

    res = np.array(res)
    res = res.transpose()
    precisions = res[0]
    recall = res[1]
    new_r = np.arange(0.,100.01,10.).tolist()
    new_p = []
    for r in new_r:
        new_p.append(np.interp(r,recall,precisions))
    precisions = np.array(new_p)
    if precisions.shape[0]>1:
        for i in range(precisions.shape[0]-1):
            precisions[i] = np.max(precisions[i+1:])
    return np.mean(precisions)


def getRecall(gtboxes,gtlabels,boxes,labels,threshold=0.5):
    gt_shape = gtboxes.shape
    #indict if there have some box match with this ground-truth box
    gt_mask = np.zeros([gt_shape[0]],dtype=np.int32)
    boxes_shape = boxes.shape
    #indict if there have some ground-truth box match with this box
    boxes_mask = np.zeros(boxes_shape[0],dtype=np.int32)
    gt_size = gtlabels.shape[0]
    boxes_size = labels.shape[0]
    for i in range(gt_size):
        max_index = -1
        max_jaccard = 0.0
        #iterator on all boxes to find one have the most maximum jacard value with current ground-truth box
        for j in range(boxes_size):
            if gtlabels[i] != labels[j] or boxes_mask[j] != 0:
                continue
            jaccard = npod.box_jaccard(gtboxes[i],boxes[j])
            if jaccard>threshold and jaccard > max_jaccard:
                max_jaccard = jaccard
                max_index = j

        if max_index < 0:
            continue

        gt_mask[i] = 1
        boxes_mask[max_index] = 1

    correct_num = np.sum(gt_mask)
    total_num = gt_size

    if 0 == total_num:
        return 100.

    return 100.*correct_num/total_num

def getPrecision(gtboxes,gtlabels,boxes,labels,threshold=0.5,auto_scale_threshold=True,ext_info=False):
    '''
    :param gtboxes: [N,4]
    :param gtlabels: [N]
    :param boxes: [M,4]
    :param labels: [M]
    :param threshold: float
    :return: precision,recall float
    '''
    if not isinstance(gtboxes,np.ndarray):
        gtboxes = np.array(gtboxes)
    if not isinstance(gtlabels,np.ndarray):
        gtlabels = np.array(gtlabels)
    gt_shape = gtboxes.shape
    #indict if there have some box match with this ground-truth box
    gt_mask = np.zeros([gt_shape[0]],dtype=np.int32)
    boxes_shape = boxes.shape
    #indict if there have some ground-truth box match with this box
    boxes_mask = np.zeros(boxes_shape[0],dtype=np.int32)
    gt_size = gtlabels.shape[0]
    boxes_size = labels.shape[0]
    MIN_VOL = 0.005
    #print(">>>>",gtboxes,gtlabels)
    for i in range(gt_size):
        max_index = -1
        max_jaccard = 0.0

        t_threshold = threshold
        if auto_scale_threshold:
            #print(i,gtboxes,gtlabels)
            vol = npod.box_vol(gtboxes[i])
            if vol < MIN_VOL:
                t_threshold = vol*threshold/MIN_VOL
        #iterator on all boxes to find one have the most maximum jacard value with current ground-truth box
        for j in range(boxes_size):
            if gtlabels[i] != labels[j] or boxes_mask[j] != 0:
                continue

            jaccard = npod.box_jaccard(gtboxes[i],boxes[j])
            if jaccard>t_threshold and jaccard > max_jaccard:
                max_jaccard = jaccard
                max_index = j

        if max_index < 0:
            continue

        gt_mask[i] = 1
        boxes_mask[max_index] = 1

    correct_num = np.sum(gt_mask)

    recall = __safe_persent(correct_num,gt_size)
    precision = __safe_persent(correct_num,boxes_size)

    if ext_info:
        gt_label_list = []
        for i in range(gt_mask.shape[0]):
            if gt_mask[i] != 1:
                gt_label_list.append(gtlabels[i])
        pred_label_list = []
        for i in range(boxes_size):
            if boxes_mask[i] != 1:
                pred_label_list.append(labels[i])
        return precision,recall,gt_label_list,pred_label_list
    else:
        return precision,recall

class ModelPerformance:
    def __init__(self,threshold,no_mAP=False,no_F1=False):
        self.total_map = 0.
        self.total_recall = 0.
        self.total_precision = 0.
        self.total_F1 = 0.
        self.threshold = threshold
        self.test_nr = 0
        self.no_mAP=no_mAP
        self.no_F1 = no_F1

    def __call__(self, gtboxes,gtlabels,boxes,labels,probability=None):
        gtboxes = copy.deepcopy(np.array(gtboxes))
        gtlabels = copy.deepcopy(np.array(gtlabels))
        boxes = copy.deepcopy(boxes)
        labels = copy.deepcopy(labels)
        if probability is not None:
            probability = copy.deepcopy(probability)

        if self.no_mAP:
            ap = 0.
        else:
            ap = getmAP(gtboxes, gtlabels, boxes, labels, probability=probability,threshold=self.threshold)

        rc = getRecall(gtboxes, gtlabels, boxes, labels, self.threshold)

        if self.no_F1:
            f1 = 0.
        else:
            f1 = getF1(gtboxes, gtlabels, boxes, labels, self.threshold)

        pc,_ = getPrecision(gtboxes, gtlabels, boxes, labels, self.threshold)

        self.total_map += ap
        self.total_recall += rc
        self.total_precision  += pc
        self.total_F1 += f1
        self.test_nr += 1
        return ap,rc,pc,f1

    @staticmethod
    def safe_div(v0,v1):
        if math.fabs(v1)<1e-8:
            return 0.
        return v0/v1

    def __getattr__(self, item):
        if item=="mAP":
            return self.safe_div(self.total_map,self.test_nr)
        elif item =="recall":
            return self.safe_div(self.total_recall,self.test_nr)
        elif item=="precision":
            return self.safe_div(self.total_precision,self.test_nr)

class COCOEvaluation(object):
    def __init__(self,categories_list=None,num_classes=None):
        if categories_list is None:
            self.categories_list = [{"id":x+1,"name":str(x+1)} for x in range(num_classes)]
        else:
            self.categories_list = categories_list
        self.coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
            self.categories_list,include_metrics_per_category=False)
        self.image_id = 0
    def __call__(self, gtboxes,gtlabels,boxes,labels,probability=None,img_size=[512,512],is_crowd=None):
        if probability is None:
            probability = np.ones_like(labels,dtype=np.float32)
        if not isinstance(gtboxes,np.ndarray):
            gtboxes = np.array(gtboxes)
        if not isinstance(gtlabels,np.ndarray):
            gtlabels = np.array(gtlabels)
        if not isinstance(boxes,np.ndarray):
            boxes = np.array(boxes)
        if not isinstance(labels,np.ndarray):
            labels = np.array(labels)
        if probability is not None and not isinstance(probability,np.ndarray):
            probability = np.array(probability)
        if gtlabels.shape[0]>0:
            gtboxes = gtboxes*[[img_size[0],img_size[1],img_size[0],img_size[1]]]
            if is_crowd is None:
                self.coco_evaluator.add_single_ground_truth_image_info(
                    image_id=str(self.image_id),
                    groundtruth_dict={
                        standard_fields.InputDataFields.groundtruth_boxes:
                            gtboxes,
                        standard_fields.InputDataFields.groundtruth_classes:gtlabels,
                    })
            else:
                if not isinstance(is_crowd,np.ndarray):
                    is_crowd = np.array(is_crowd)
                self.coco_evaluator.add_single_ground_truth_image_info(
                    image_id=str(self.image_id),
                    groundtruth_dict={
                        standard_fields.InputDataFields.groundtruth_boxes:
                            gtboxes,
                        standard_fields.InputDataFields.groundtruth_classes:gtlabels,
                        standard_fields.InputDataFields.groundtruth_is_crowd:
                            is_crowd
                    })
        if labels.shape[0]>0 and gtlabels.shape[0]>0:
            boxes = boxes*[[img_size[0],img_size[1],img_size[0],img_size[1]]]
            self.coco_evaluator.add_single_detected_image_info(
                image_id=str(self.image_id),
                detections_dict={
                    standard_fields.DetectionResultFields.detection_boxes:
                        boxes,
                    standard_fields.DetectionResultFields.detection_scores:
                        probability,
                    standard_fields.DetectionResultFields.detection_classes:
                        labels
                })
        self.image_id += 1

    def evaluate(self):
        print(f"Test size {len(self.coco_evaluator._image_ids)}")
        return self.coco_evaluator.evaluate()
    def show(self):
        print(f"Test size {len(self.coco_evaluator._image_ids)}")
        self.coco_evaluator.evaluate()

class ClassesWiseModelPerformace(object):
    def __init__(self,num_classes,threshold=0.5,classes_begin_value=1,model_type=COCOEvaluation,model_args={}):
        self.num_classes = num_classes
        self.clases_begin_value = classes_begin_value
        self.data = []
        for i in range(self.num_classes):
            self.data.append(model_type(num_classes=num_classes,**model_args))
        self.mp = model_type(num_classes=num_classes,**model_args)

    @staticmethod
    def select_bboxes_and_labels(bboxes,labels,classes):
        if len(labels) == 0:
            return np.array([]),np.array([])
        mask = np.equal(labels,classes)
        rbboxes = bboxes[mask,:]
        rlabels = labels[mask]
        return rbboxes,rlabels

    def __call__(self, gtboxes,gtlabels,boxes,labels,probability=None):
        if not isinstance(gtboxes,np.ndarray):
            gtboxes = np.array(gtboxes)
        if not isinstance(gtlabels,np.ndarray):
            gtlabels = np.array(gtlabels)

        for i in range(self.num_classes):
            classes = i+self.clases_begin_value
            lgtboxes,lgtlabels = self.select_bboxes_and_labels(gtboxes,gtlabels,classes)
            lboxes,llabels = self.select_bboxes_and_labels(boxes,labels,classes)
            if lgtlabels.shape[0]==0:
                continue
            self.data[i](lgtboxes,lgtlabels,lboxes,llabels)
        return self.mp(gtboxes,gtlabels,boxes,labels)

    def show(self):
        for i in range(self.num_classes):
            classes = i+self.clases_begin_value
            print(f"Classes:{classes}")
            self.data[i].show()

    def __getattr__(self, item):
        if item=="mAP":
            return self.mp.mAP
        elif item =="recall":
            return self.mp.recall
        elif item=="precision":
            return self.mp.precision



def removeLabels(bboxes,labels,labels_to_remove):
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes)
    if not isinstance(labels,np.ndarray):
        labels = np.array(labels)
    for l in labels_to_remove:
        bboxes,labels = removeLabel(bboxes,labels,l)
    return bboxes,labels

def removeLabel(bboxes,labels,label_to_remove):
    if(bboxes.shape[0] == 0):
        return bboxes,labels
    mask = np.not_equal(labels,label_to_remove)
    return bboxes[mask,:],labels[mask]

'''
use the object's probability, calculate background's probability.
example: objcet's probability is p, then background's probability is 1-p
probibality:[X]目标的概率，
labels:[X]目标的类别
num_classes:scale,目标类别数（包括背景）
输出[X,num_classes], 其中res[,0]为1-probibality
'''
def get_total_probibality_by_object_probibality(probibality,labels,num_classes):
    #negnative object's probability
    n_prob = 1.0-probibality
    shape = tf.shape(probibality)
    index = tf.range(shape[0])
    index0 = tf.stack([index,labels],axis=1)
    index1 = tf.stack([index,tf.zeros_like(labels)],axis=1)
    v0 = tf.sparse_to_dense(sparse_indices=index0,sparse_values=probibality,output_shape=[shape[0],num_classes])
    v1 = tf.sparse_to_dense(sparse_indices=index1,sparse_values=n_prob,output_shape=[shape[0],num_classes])
    return v0+v1

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s %(funcName)s:%(message)s',
                        datefmt="%H:%M:%S")
    boxes = []
    labels = []
    gtboxes = []
    gtlabels = []
    probability = []
    for i in range(10):
        gtboxes.append([i,i,i+1,i+1])
        boxes.append([i,i,i+1,i+1])
        labels.append(i%4)
        gtlabels.append(i%4)
        probability.append(i)
    probability.reverse()
    for i in range(5):
        boxes[i+5][2] += 1


    print(getmAP(gtboxes=gtboxes,gtlabels=gtlabels,boxes=boxes,labels=labels,probability=probability))
