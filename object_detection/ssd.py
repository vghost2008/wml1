#coding=utf-8
import tensorflow as tf
from abc import ABCMeta, abstractmethod
import object_detection.bboxes as bboxes
from wtfop.wtfop_ops import boxes_encode
import object_detection.losses as losses
import numpy as np
import object_detection.architectures_tools as atools
import object_detection.od_toolkit as od
import math

slim = tf.contrib.slim

class SSD(object):
    __metaclass__ = ABCMeta
    def __init__(self,num_classes,batch_size=4,pred_bboxes_classwise=False):
        self.scales=[]
        self.ratios=[]
        self.anchors= None
        self.np_anchors=[]
        self.inputs=None
        self.batch_size = batch_size
        self.feature_maps_shape = []
        self.num_classes = num_classes
        '''
        回归损失乘以reg_loss_weight以增加回归的准确率
        '''
        self.reg_loss_weight = 3.
        self.pred_bboxes_classwise = pred_bboxes_classwise
        self.logits = None
        self.regs = None

    def getAnchorBoxes(self):
        np_anchors=[]
        for i,shape in enumerate(self.feature_maps_shape):
            anchors = bboxes.get_anchor_bboxes(shape,sizes=self.scales[i],ratios=self.ratios[i],is_area=False)
            np_anchors.append(anchors)

        np_anchors = np.concatenate(np_anchors,axis=0)
        anchors = tf.convert_to_tensor(np_anchors)
        self.anchors = tf.expand_dims(anchors,axis=0)
        return self.anchors

    def encodeBoxes(self,gbboxes, glabels,lens,pos_threshold=0.7,neg_threshold=0.3):
        '''
        :param gbboxes:
        :param glabels:
        :param lens:
        :param pos_threshold:
        :param neg_threshold:
        :return:
        '''
        #anchors can be [1,X,4] or [batch_size,X,4], usually use [1,X,4]
        with tf.name_scope("EncodeBoxes"):
            gtregs, gtlabels, gtscores,remove_indices = boxes_encode(bboxes=self.anchors,
                                                                   gboxes=gbboxes,
                                                                   glabels=glabels,
                                                                   length=lens,
                                                                   pos_threshold=pos_threshold,
                                                                   neg_threshold=neg_threshold)
            self.gtregs = gtregs
            self.gtlabels = gtlabels
            self.gtscores = gtscores
            self.anchor_remove_indices = remove_indices
        return gtregs, gtlabels, gtscores,remove_indices

    def getLoss(self,use_focal_loss=True):
        return losses.od_loss(gregs=self.gtregs,
                   glabels=self.gtlabels,
                   classes_logits=self.logits,
                   bboxes_regs=self.regs,
                   num_classes=self.num_classes,
                   reg_loss_weight=self.reg_loss_weight,
                   bboxes_remove_indices=self.anchor_remove_indices,
                   scope="Loss",
                   classes_wise=self.pred_bboxes_classwise,
                   use_focal_loss=use_focal_loss)

    def merge_classes_predictor(self,logits):
        '''
        :param logits: list of [batch_size,hi,wi,channel] tensor, hi,wi,channel must fully defined
        :return:
        [batch_size,X,num_classes]
        note: channel should be num_classes*len(scales[i])*len(ratios[i])
        '''
        logits_list = []
        for lg in logits:
            shape = lg.get_shape().as_list()
            x_size = shape[1]*shape[2]*shape[3]//self.num_classes
            lg = tf.reshape(lg,[-1,x_size,self.num_classes])
            logits_list.append(lg)
        return tf.concat(logits_list,axis=1)

    def merge_box_predictor(self,regs):
        '''
        :param logits: list of [batch_size,hi,wi,channel] tensor, hi,wi,channel must fully defined
        :return:
        [batch_size,X,4]
        note: channel should be 4*len(scales[i])*len(ratios[i]) or 4*num_classes*len(scales[i])*len(ratios[i])
        if self.pred_bboxes_classwise is True
        '''
        regs_list = []
        for rg in regs:
            shape = rg.get_shape().as_list()
            x_size = shape[1]*shape[2]*shape[3]//4
            rg = tf.reshape(rg,[-1,x_size,4])
            regs_list.append(rg)
        return tf.concat(regs_list,axis=1)

    @abstractmethod
    def buildNet(self,inputs):
        pass

    def buildPredictor(self,feature_maps):
        logits_list = []
        boxes_regs_list = []
        for i,net in enumerate(feature_maps.values()):
            with tf.variable_scope(f"BoxPredictor_{i}"):
                logits_nr = len(self.scales[i])*len(self.ratios[i])
                if self.pred_bboxes_classwise:
                    regs_nr = logits_nr*self.num_classes
                else:
                    regs_nr = logits_nr
                logits_net = slim.conv2d(net,logits_nr*self.num_classes, [1, 1], activation_fn=None,
                                         normalizer_fn=None,
                                         scope="ClassPredictor")
                boxes_regs = slim.conv2d(net, regs_nr*4, [1, 1], activation_fn=None,
                                         normalizer_fn=None,
                                         scope="BoxEncodingPredictor")

                logits_list.append(logits_net)
                boxes_regs_list.append(boxes_regs)

        self.logits = self.merge_classes_predictor(logits_list)
        self.regs = self.merge_box_predictor(boxes_regs_list)
        return self.logits,self.regs

    @staticmethod
    def multi_resolution_feature_maps(feature_map_layout, depth_multiplier,
                                  min_depth, insert_1x1_conv, image_features,
                                  pool_residual=False):
        return atools.multi_resolution_feature_maps(feature_map_layout, depth_multiplier,
                                  min_depth, insert_1x1_conv, image_features,
                                  pool_residual)
    @staticmethod
    def fpn_top_down_feature_maps(*args,**kwargs):
        return atools.fpn_top_down_feature_maps(*args,**kwargs)

    '''
    min_scale: min scale (edge size)
    max_scale: max scale (edge size)
    last_keep_one: where the last stage only keep one scale
    return: 
    scale's list, from min to max
    '''
    def get_scales(self,min_scale=0.2,max_scale=0.9,fm_nr=5,last_keep_one=True):
        def scales(stage):
            return max_scale-(fm_nr-1-stage)*(max_scale-min_scale)/(fm_nr-1)
        for i in range(fm_nr):
            self.scales.append([scales(i),math.sqrt(scales(i)*scales(i+1))])
        if last_keep_one:
            self.scales[-1] = [self.scales[-1][0]]

        return self.scales

    def getBoxes(self,k=1000,threshold=0.5,nms_threshold=0.1,
                 limits=None,
                 classes_wise_nms=True,
                 use_soft_nms=True):
        with tf.variable_scope("GetBoxes"):
            probs = tf.nn.softmax(self.logits)
            self.boxes,self.labels,self.probs,self.indices,self.boxes_lens = \
            od.get_predictionv2(
                class_prediction=probs,
                bboxes_regs=self.regs,
                proposal_bboxes=self.anchors,
                threshold=threshold,
                nms_threshold=nms_threshold,
                limits=limits,
                candiate_nr=k,
                classes_wise=self.pred_bboxes_classwise,
                classes_wise_nms=classes_wise_nms,
                use_soft_nms=use_soft_nms)
        return self.boxes,self.labels,self.probs,self.boxes_lens

