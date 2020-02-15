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
from thirdparty.odmetrics import coco_evaluation
from thirdparty.odmetrics import standard_fields
import wml_utils as wmlu
import img_utils as wmli
import copy



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
    def __init__(self,categories_list=None,num_classes=None,mask_on=False):
        if categories_list is None:
            self.categories_list = [{"id":x+1,"name":str(x+1)} for x in range(num_classes)]
        else:
            self.categories_list = categories_list
        if not mask_on:
            self.coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
                self.categories_list,include_metrics_per_category=False)
        else:
            self.coco_evaluator = coco_evaluation.CocoMaskEvaluator(
                self.categories_list,include_metrics_per_category=False)
        self.image_id = 0
    '''
    gtboxes:[N,4]
    gtlabels:[N]
    img_size:[H,W]
    gtmasks:[N,H,W]
    '''
    def __call__(self, gtboxes,gtlabels,boxes,labels,probability=None,img_size=[512,512],
                 gtmasks=None,
                 masks=None,is_crowd=None):
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
            groundtruth_dict={
                standard_fields.InputDataFields.groundtruth_boxes:
                    gtboxes,
                standard_fields.InputDataFields.groundtruth_classes:gtlabels,
            }
            if is_crowd is not None:
                if not isinstance(is_crowd,np.ndarray):
                    is_crowd = np.array(is_crowd)
                groundtruth_dict[standard_fields.InputDataFields.groundtruth_is_crowd] = is_crowd
            if gtmasks is not None:
                groundtruth_dict[standard_fields.InputDataFields.groundtruth_instance_masks] = gtmasks
            self.coco_evaluator.add_single_ground_truth_image_info(
                image_id=str(self.image_id),
                groundtruth_dict=groundtruth_dict)
        if labels.shape[0]>0 and gtlabels.shape[0]>0:
            boxes = boxes*[[img_size[0],img_size[1],img_size[0],img_size[1]]]
            detections_dict={
                standard_fields.DetectionResultFields.detection_boxes:
                    boxes,
                standard_fields.DetectionResultFields.detection_scores:
                    probability,
                standard_fields.DetectionResultFields.detection_classes:
                    labels
            }
            if masks is not None:
                detections_dict[standard_fields.DetectionResultFields.detection_masks] = masks
            self.coco_evaluator.add_single_detected_image_info(
                image_id=str(self.image_id),
                detections_dict=detections_dict)
        self.image_id += 1

    def num_examples(self):
        if '_image_ids_with_detections' in self.coco_evaluator.__dict__:
            return len(self.coco_evaluator._image_ids_with_detections)
        elif '_image_ids' in self.coco_evaluator.__dict__:
            return len(self.coco_evaluator._image_ids)
        else:
            raise RuntimeError("Error evaluator type.")

    def evaluate(self):
        print(f"Test size {self.num_examples()}")
        return self.coco_evaluator.evaluate()
    def show(self):
        print(f"Test size {self.num_examples()}")
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

