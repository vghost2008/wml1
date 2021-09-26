#coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import random
import os
import object_detection2.npod_toolkit as npod
import math
from semantic.visualization_utils import draw_bounding_boxes_on_image_tensors
import object_detection2.visualization as odv
import wml_utils
import logging
import shutil
from thirdparty.odmetrics import coco_evaluation
from thirdparty.odmetrics import standard_fields
import wml_utils as wmlu
import img_utils as wmli
import copy
from collections import OrderedDict


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
        #iterator on all boxes to find one which have the most maximum jacard value with current ground-truth box
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
return:
mAP:[0,100]
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
    P_v = gt_size
    TP_v = correct_num
    FP_v = boxes_size-correct_num


    if ext_info:
        gt_label_list = []
        for i in range(gt_mask.shape[0]):
            if gt_mask[i] != 1:
                gt_label_list.append(gtlabels[i])
        pred_label_list = []
        for i in range(boxes_size):
            if boxes_mask[i] != 1:
                pred_label_list.append(labels[i])
        return precision,recall,gt_label_list,pred_label_list,TP_v,FP_v,P_v
    else:
        return precision,recall

class PrecisionAndRecall:
    def __init__(self,threshold=0.5,num_classes=90,label_trans=None,*args,**kwargs):
        self.threshold = threshold
        self.gtboxes = []
        self.gtlabels = []
        self.boxes = []
        self.labels = []
        self.precision = None
        self.recall = None
        self.total_test_nr = 0
        self.num_classes = num_classes
        self.label_trans = label_trans

    def __call__(self, gtboxes,gtlabels,boxes,labels,probability=None,img_size=[512,512],
                 gtmasks=None,
                 masks=None,is_crowd=None):
        if self.label_trans is not None:
            gtlabels = self.label_trans(gtlabels)
            labels = self.label_trans(labels)
        if gtboxes.shape[0]>0:
            self.gtboxes.append(gtboxes)
            self.gtlabels.append(np.array(gtlabels)+self.total_test_nr*self.num_classes)
        if boxes.shape[0]>0:
            self.boxes.append(boxes)
            self.labels.append(np.array(labels)+self.total_test_nr*self.num_classes)
        self.total_test_nr += 1

    def evaluate(self):
        if self.total_test_nr==0 or len(self.boxes)==0 or len(self.labels)==0:
            self.precision,self.recall = 0,0
            return
        gtboxes = np.concatenate(self.gtboxes,axis=0)
        gtlabels = np.concatenate(self.gtlabels,axis=0)
        boxes = np.concatenate(self.boxes,axis=0)
        labels = np.concatenate(self.labels,axis=0)
        self.precision,self.recall = getPrecision(gtboxes, gtlabels, boxes, labels, threshold=self.threshold,
                                                  auto_scale_threshold=False, ext_info=False)
    def show(self,name=""):
        self.evaluate()
        res = f"{name}: total test nr {self.total_test_nr}, precision {self.precision:.3f}, recall {self.recall:.3f}"
        print(res)

    def to_string(self):
        return f"{self.precision:.3f}/{self.recall:.3f}({self.total_test_nr})"

class ROC:
    def __init__(self,threshold=0.5,num_classes=90,label_trans=None,*args,**kwargs):
        self.threshold = threshold
        self.gtboxes = []
        self.gtlabels = []
        self.boxes = []
        self.labels = []
        self.probs = []
        self.precision = None
        self.recall = None
        self.total_test_nr = 0
        self.num_classes = num_classes
        self.label_trans = label_trans
        self.results = None

    def __call__(self, gtboxes,gtlabels,boxes,labels,probability=None,img_size=[512,512],
                 gtmasks=None,
                 masks=None,is_crowd=None):
        if self.label_trans is not None:
            gtlabels = self.label_trans(gtlabels)
            labels = self.label_trans(labels)
        if gtboxes.shape[0]>0:
            self.gtboxes.append(gtboxes)
            self.gtlabels.append(np.array(gtlabels)+self.total_test_nr*self.num_classes)
        if boxes.shape[0]>0:
            self.boxes.append(boxes)
            self.labels.append(np.array(labels)+self.total_test_nr*self.num_classes)
            self.probs.append(np.array(probability))
        self.total_test_nr += 1

    def evaluate(self):
        if self.total_test_nr==0 or len(self.boxes)==0 or len(self.labels)==0:
            self.precision,self.recall = 0,0
            return
        gtboxes = np.concatenate(self.gtboxes,axis=0)
        gtlabels = np.concatenate(self.gtlabels,axis=0)
        boxes = np.concatenate(self.boxes,axis=0)
        labels = np.concatenate(self.labels,axis=0)
        probs = np.concatenate(self.probs,axis=0)
        self.results = []

        for p in np.arange(0,1,0.05):
            mask = np.greater(probs,p)
            t_boxes = boxes[mask]
            t_labels = labels[mask]
            precision, recall, gt_label_list, pred_label_list, TP_v, FP_v, P_v = \
                getPrecision(gtboxes, gtlabels, t_boxes, t_labels, threshold=self.threshold,
                                                  auto_scale_threshold=False, ext_info=True)
            self.results.append([p,precision,recall])

    def show(self,name=""):
        print(self.to_string())

    def to_string(self):
        self.evaluate()
        res = ""
        if self.results is None or len(self.results) == 0:
            return res
        for p, precision, recall in self.results:
            res += f"{p:.3f},{precision:.3f},{recall:.3f};\n"

        return res

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

class GeneralCOCOEvaluation(object):
    def __init__(self,categories_list=None,num_classes=None,mask_on=False,label_trans=None):
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
        self.label_trans = label_trans
        self.image_id = 0
        self.cached_values = {}
    '''
    gtboxes:[N,4]
    gtlabels:[N]
    img_size:[H,W]
    gtmasks:[N,H,W]
    '''
    def __call__(self, gtboxes,gtlabels,boxes,labels,probability=None,img_size=[512,512],
                 gtmasks=None,
                 masks=None,is_crowd=None,use_relative_coord=True):
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
        if self.label_trans is not None:
            gtlabels = self.label_trans(gtlabels)
            labels = self.label_trans(labels)
        if probability is not None and not isinstance(probability,np.ndarray):
            probability = np.array(probability)
        if gtlabels.shape[0]>0:
            if use_relative_coord:
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
            if use_relative_coord:
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

    def show(self,name=""):
        print(f"Test size {self.num_examples()}")
        res = self.coco_evaluator.evaluate()
        str0 = "|配置|"
        str1 = "|---|"
        str2 = f"|{name}|"
        for k,v in res.items():
            index = k.find("/")
            if index>0:
                k = k[index+1:]
            self.cached_values[k] = v
            str0 += f"{k}|"
            str1 += "---|"
            str2 += f"{v:.3f}|"
        print(str0)
        print(str1)
        print(str2)
        return res

    def to_string(self):
        if 'mAP' in self.cached_values and 'mAP@.50IOU' in self.cached_values:
            return f"{self.cached_values['mAP']:.3f}/{self.cached_values['mAP@.50IOU']:.3f}"
        else:
            return f"N.A."

class COCOBoxEvaluation(GeneralCOCOEvaluation):
    def __init__(self,categories_list=None,num_classes=None,label_trans=None):
        super().__init__(categories_list=categories_list,
                         num_classes=num_classes,
                         mask_on=False,
                         label_trans=label_trans)

class COCOMaskEvaluation(GeneralCOCOEvaluation):
    def __init__(self,categories_list=None,num_classes=None,label_trans=None):
        super().__init__(categories_list=categories_list,
                         num_classes=num_classes,
                         mask_on=True,
                         label_trans=label_trans)


class COCOEvaluation(object):
    '''
    num_classes: 不包含背景 
    '''
    def __init__(self,categories_list=None,num_classes=None,mask_on=False,label_trans=None):
        self.box_evaluator = COCOBoxEvaluation(categories_list=categories_list,
                                               num_classes=num_classes,
                                               label_trans=label_trans)
        self.mask_evaluator = None
        if mask_on:
            self.mask_evaluator = COCOMaskEvaluation(categories_list=categories_list,
                                                     num_classes=num_classes,
                                                     label_trans=label_trans)
    def __call__(self, *args, **kwargs):
        self.box_evaluator(*args,**kwargs)
        if self.mask_evaluator is not None:
            self.mask_evaluator(*args,**kwargs)

    def num_examples(self):
        return self.box_evaluator.num_examples()

    def evaluate(self):
        res = self.box_evaluator.evaluate()
        if self.mask_evaluator is not None:
            res1 = self.mask_evaluator.evaluate()
            return res,res1
        return res

    def show(self,name=""):
        self.box_evaluator.show(name=name)
        if self.mask_evaluator is not None:
            self.mask_evaluator.show(name=name)

    def to_string(self):
        if self.mask_evaluator is not None:
            return self.box_evaluator.to_string()+";"+self.mask_evaluator.to_string()
        else:
            return self.box_evaluator.to_string()



class ClassesWiseModelPerformace(object):
    def __init__(self,num_classes,threshold=0.5,classes_begin_value=1,model_type=COCOEvaluation,model_args={},label_trans=None,
                 **kwargs):
        self.num_classes = num_classes
        self.clases_begin_value = classes_begin_value
        self.data = []
        for i in range(self.num_classes):
            self.data.append(model_type(num_classes=num_classes,**model_args))
        self.mp = model_type(num_classes=num_classes,**model_args)
        self.label_trans = label_trans
        self.have_data = np.zeros([num_classes],dtype=np.bool)

    @staticmethod
    def select_bboxes_and_labels(bboxes,labels,classes):
        if len(labels) == 0:
            return np.array([],dtype=np.float32),np.array([],dtype=np.int32),np.array([],dtype=np.bool)
        if not isinstance(labels,np.ndarray):
            labels = np.array(labels)
        mask = np.equal(labels,classes)
        rbboxes = bboxes[mask,:]
        rlabels = labels[mask]
        return rbboxes,rlabels,mask

    def __call__(self, gtboxes,gtlabels,boxes,labels,probability=None,img_size=None,use_relative_coord=True):
        if not isinstance(gtboxes,np.ndarray):
            gtboxes = np.array(gtboxes)
        if not isinstance(gtlabels,np.ndarray):
            gtlabels = np.array(gtlabels)
        if not isinstance(labels,np.ndarray):
            labels = np.array(labels)
        if self.label_trans is not None:
            gtlabels = self.label_trans(gtlabels)
            labels = self.label_trans(labels)
            
        for i in range(self.num_classes):
            classes = i+self.clases_begin_value
            lgtboxes,lgtlabels,_ = self.select_bboxes_and_labels(gtboxes,gtlabels,classes)
            lboxes,llabels,lmask = self.select_bboxes_and_labels(boxes,labels,classes)
            if probability is not None:
                lprobs = probability[lmask]
            else:
                lprobs = None
            if lgtlabels.shape[0]==0:
                continue
            self.have_data[i] = True
            self.data[i](lgtboxes,lgtlabels,lboxes,llabels,lprobs,img_size=img_size,use_relative_coord=use_relative_coord)
        return self.mp(gtboxes,gtlabels,boxes,labels)

    def show(self):
        for i in range(self.num_classes):
            if not self.have_data[i]:
                continue
            classes = i+self.clases_begin_value
            print(f"Classes:{classes}")
            self.data[i].show()
        self.mp.show()
        str0 = "|配置|"
        str1 = "|---|"
        str2 = "||"
        for i in range(self.num_classes):
            str0 += f"C{i+1}|"
            str1 += "---|"
            str2 += f"{str(self.data[i].to_string())}|"
        print(str0)
        print(str1)
        print(str2)

    def __getattr__(self, item):
        if item=="mAP":
            return self.mp.mAP
        elif item =="recall":
            return self.mp.recall
        elif item=="precision":
            return self.mp.precision

class SubsetsModelPerformace(object):
    def __init__(self, num_classes, sub_sets,threshold=0.5, model_type=COCOEvaluation, model_args={},
                 label_trans=None,
                 **kwargs):
        '''

        :param num_classes: 不包含背景
        :param sub_sets: list(list):如[[1,2],[3,4,5]]表示label 1,2一组进行评估,label 3 ,4,5一组进行评估
        :param threshold:
        :param classes_begin_value:
        :param model_type:
        :param model_args:
        :param label_trans:
        '''
        self.num_classes = num_classes
        self.data = []
        self.sub_sets = sub_sets
        for i in range(len(sub_sets)):
            self.data.append(model_type(num_classes=num_classes, **model_args))
        self.mp = model_type(num_classes=num_classes, **model_args)
        self.label_trans = label_trans

    @staticmethod
    def select_bboxes_and_labels(bboxes, labels, classes):
        if len(labels) == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int32), np.array([], dtype=np.bool)

        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        mask = np.zeros_like(labels, dtype=np.bool)
        for i,l in enumerate(labels):
            if l in classes:
                mask[i] = True
        rbboxes = bboxes[mask, :]
        rlabels = labels[mask]
        return rbboxes, rlabels,mask

    def __call__(self, gtboxes, gtlabels, boxes, labels, probability=None, img_size=None):
        if not isinstance(gtboxes, np.ndarray):
            gtboxes = np.array(gtboxes)
        if not isinstance(gtlabels, np.ndarray):
            gtlabels = np.array(gtlabels)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        if self.label_trans is not None:
            gtlabels = self.label_trans(gtlabels)
            labels = self.label_trans(labels)

        for i,sub_set_labels in enumerate(self.sub_sets):
            lgtboxes, lgtlabels,_ = self.select_bboxes_and_labels(gtboxes, gtlabels, sub_set_labels)
            lboxes, llabels,lmask = self.select_bboxes_and_labels(boxes, labels, sub_set_labels)
            if probability is not None:
                lprobs = probability[lmask]
            else:
                lprobs = None
            if lgtlabels.shape[0] == 0:
                continue
            self.data[i](lgtboxes, lgtlabels, lboxes, llabels, lprobs,img_size=img_size)
        return self.mp(gtboxes, gtlabels, boxes, labels)

    def show(self):
        for i,sub_set_labels in enumerate(self.sub_sets):
            print(f"Classes:{sub_set_labels}")
            self.data[i].show()
        str0 = "|配置|"
        str1 = "|---|"
        str2 = "||"
        for i,sub_set_labels in enumerate(self.sub_sets):
            str0 += f"S{sub_set_labels}|"
            str1 += "---|"
            str2 += f"{str(self.data[i].to_string())}|"
        print(str0)
        print(str1)
        print(str2)

    def __getattr__(self, item):
        if item == "mAP":
            return self.mp.mAP
        elif item == "recall":
            return self.mp.recall
        elif item == "precision":
            return self.mp.precision


class  MeanIOU(object):
    def __init__(self,num_classes,*args,**kwargs):
        self.intersection = np.zeros(shape=[num_classes],dtype=np.int64)
        self.union = np.zeros(shape=[num_classes],dtype=np.int64)
        self.num_classes = num_classes

    def get_per_classes_iou(self):
        return self.intersection/np.maximum(self.union,1e-8)

    def get_mean_iou(self):
        return np.mean(self.get_per_classes_iou())
        
    
    def __call__(self, gtlabels,predictions):
        all_equal = np.equal(gtlabels,predictions)
        for i in range(1,self.num_classes+1):
            mask = np.equal(gtlabels,i)
            t_int = np.sum(all_equal[mask].astype(np.int64))
            t_data0 = np.sum(np.equal(gtlabels,i).astype(np.int64))
            t_data1 = np.sum(np.equal(predictions,i).astype(np.int64))
            t_union = t_data0+t_data1-t_int
            self.intersection[i-1] += t_int
            self.union[i-1] += t_union


    def show(self,name):
        str0 = "|配置|mIOU|"
        str1 = "|---|---|"
        str2 = f"|{name}|{self.get_mean_iou():.4f}"
        data = self.get_per_classes_iou()
        for i in range(self.num_classes):
            str0 += f"C{i+1}|"
            str1 += "---|"
            str2 += f"{data[i]:.3f}|"
        print(str0)
        print(str1)
        print(str2)
