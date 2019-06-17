#coding=utf-8
import tensorflow as tf
from abc import ABCMeta, abstractmethod
import object_detection.bboxes as bboxes
from wtfop.wtfop_ops import boxes_encode,multi_anchor_generator,anchor_generator
import object_detection.losses as losses
import numpy as np
import object_detection.architectures_tools as atools
import object_detection.od_toolkit as od
import math
import functools
import wtfop.wtfop_ops as wop
import wml_utils as wmlu
import wnnlayer as wnnl
import wml_tfutils as wmlt

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
        '''
        list of [h,w]
        '''
        self.feature_maps_shape = []
        self.num_classes = num_classes
        '''
        回归损失乘以reg_loss_weight以增加回归的准确率
        '''
        self.reg_loss_weight = 3.
        self.pred_bboxes_classwise = pred_bboxes_classwise
        self.logits = None
        self.regs = None
        self.input_img = None
        self.box_specs_list = None
        self.score_converter = tf.nn.softmax
        self.raw_probs = None

    def getAnchorBoxes(self):
        np_anchors=[]
        offset = 0
        for i,shape in enumerate(self.feature_maps_shape):
            anchors = bboxes.get_anchor_bboxes(shape,sizes=self.scales[i],ratios=self.ratios[i],is_area=False)
            print("Anchors size:",anchors.shape[0],", offset=",offset)
            offset += anchors.shape[0]
            np_anchors.append(anchors)

        np_anchors = np.concatenate(np_anchors,axis=0)
        self.np_anchors = np_anchors
        anchors = tf.convert_to_tensor(np_anchors)
        self.anchors = tf.expand_dims(anchors,axis=0)
        return self.anchors
    
    def getAnchorBoxesV2(self):
        tf_anchors=[]
        for i,shape in enumerate(self.feature_maps_shape):
            anchors = anchor_generator(shape=shape, size=tf.shape(self.input_img)[1:3], scales=self.scales[i], aspect_ratios=self.ratios[i])
            tf_anchors.append(anchors)

        anchors = tf.concat(tf_anchors,axis=0)
        self.anchors = tf.expand_dims(anchors,axis=0)
        return self.anchors

    def getAnchorBoxesByFeaturesMapV2(self,features_map):
        shapes = [tf.shape(fm)[1:3] for fm in features_map]
        self.feature_maps_shape = shapes
        return self.getAnchorBoxesV2()
    
    def getAnchorBoxesV3(self,min_scale=0.2,max_scale=0.95,
                       scales=None,
                       aspect_ratios=(1.0, 2.0, 3.0, 1.0 / 2, 1.0 / 3),
                       interpolated_scale_aspect_ratio=1.0,
                       reduce_boxes_in_lowest_layer=True,
                       size=[1,1]):
        num_layers = len(self.feature_maps_shape)
        box_specs_list = []
        if scales is None or not scales:
            scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
                      for i in range(num_layers)] + [1.0]
        else:
            # Add 1.0 to the end, which will only be used in scale_next below and used
            # for computing an interpolated scale for the largest scale in the list.
            scales += [1.0]

        tf_anchors=[]
        for layer, scale, scale_next,shape in zip(
                range(num_layers), scales[:-1], scales[1:],self.feature_maps_shape):
            layer_box_specs = []
            if layer == 0 and reduce_boxes_in_lowest_layer:
                layer_box_specs = [(0.1, 1.0), (scale, 2.0), (scale, 0.5)]
            else:
                for aspect_ratio in aspect_ratios:
                    layer_box_specs.append((scale, aspect_ratio))
                # Add one more anchor, with a scale between the current scale, and the
                # scale for the next layer, with a specified aspect ratio (1.0 by
                # default).
                if interpolated_scale_aspect_ratio > 0.0:
                    layer_box_specs.append((np.sqrt(scale * scale_next),
                                            interpolated_scale_aspect_ratio))
            box_specs_list.append(layer_box_specs)

            tf_anchors.append(SSD.get_a_layer_anchors(layer_box_specs=layer_box_specs,shape=shape,size=size))
        wmlu.show_list(box_specs_list)
        self.box_specs_list = box_specs_list
        anchors = tf.concat(tf_anchors,axis=0)
        self.anchors = tf.expand_dims(anchors,axis=0)
        return self.anchors

    @staticmethod
    def get_a_layer_anchors(layer_box_specs,shape,size):
        scales,ratios = zip(*layer_box_specs)
        return multi_anchor_generator(shape=shape,size=size,scales=scales,aspect_ratios=ratios)

    def getAnchorBoxesByFeaturesMapV3(self,features_map,min_scale=0.2,max_scale=0.95,
                       scales=None,
                       aspect_ratios=(1.0, 2.0, 3.0, 1.0 / 2, 1.0 / 3),
                       interpolated_scale_aspect_ratio=1.0,
                       reduce_boxes_in_lowest_layer=True,
                       size=[1,1]):
        shapes = [tf.shape(fm)[1:3] for fm in features_map]
        self.feature_maps_shape = shapes
        return self.getAnchorBoxesV3(min_scale=min_scale,max_scale=max_scale,
                       scales=scales,
                       aspect_ratios=aspect_ratios,
                       interpolated_scale_aspect_ratio=interpolated_scale_aspect_ratio,
                       reduce_boxes_in_lowest_layer=reduce_boxes_in_lowest_layer,size=size)

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
            gtregs, gtlabels, gtscores,remove_indices,_ = boxes_encode(bboxes=self.anchors,
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

    '''
    ssd不适合使用scores
    '''
    def getLoss(self,loss=None):
        if loss is None:
            loss = losses.ODLoss(num_classes=self.num_classes, reg_loss_weight=1.0, neg_multiplier=5.0,
                                   do_sample=True,
                                   sample_type=losses.ODLoss.SAMPLE_TYPE_BY_BAD_ORDER,
                                   sample_size=384)
        return loss(gregs=self.gtregs,
                   glabels=self.gtlabels,
                   classes_logits=self.logits,
                   bboxes_regs=self.regs,
                   bboxes_remove_indices=self.anchor_remove_indices)

    def merge_classes_predictor(self,logits,num_classes=None):
        '''
        :param logits: list of [batch_size,hi,wi,channel] tensor, hi,wi,channel must fully defined
        :return:
        [batch_size,X,num_classes]
        note: channel should be num_classes*len(scales[i])*len(ratios[i])
        '''
        if num_classes is None:
            num_classes = self.num_classes
        logits_list = []
        for lg in logits:
            shape = lg.get_shape().as_list()
            if shape[0] is None:
                x_size = shape[1]*shape[2]*shape[3]//num_classes
                lg = tf.reshape(lg,[-1,x_size,num_classes])
            else:
                lg = tf.reshape(lg,[shape[0],-1,num_classes])
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
            if shape[0] is None:
                x_size = shape[1]*shape[2]*shape[3]//4
                rg = tf.reshape(rg,[-1,x_size,4])
            else:
                rg = tf.reshape(rg,[shape[0],-1,4])
            regs_list.append(rg)
        return tf.concat(regs_list,axis=1)

    def buildNet(self,inputs,reuse=False):
        self.input_img = inputs
        return self._buildNet(inputs,reuse=reuse)
    
    @abstractmethod
    def _buildNet(self,inputs,reuse=False):
        pass

    def buildPredictor(self,feature_maps,kernel_size=[1,1]):
        logits_list = []
        boxes_regs_list = []
        for i,net in enumerate(feature_maps.values()):
            with tf.variable_scope(f"BoxPredictor_{i}"):
                if self.box_specs_list is None:
                    logits_nr = len(self.scales[i])*len(self.ratios[i])
                else:
                    logits_nr = len(self.box_specs_list[i])
                if self.pred_bboxes_classwise:
                    regs_nr = logits_nr*self.num_classes
                else:
                    regs_nr = logits_nr
                logits_net = slim.conv2d(net,logits_nr*self.num_classes, kernel_size, activation_fn=None,
                                         normalizer_fn=None,
                                         scope="ClassPredictor")
                boxes_regs = slim.conv2d(net, regs_nr*4, kernel_size, activation_fn=None,
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

    def getBoxes(self,k=1000,threshold=0.5,
                   limits=None,
                   nms=None):
        return self.getBoxesV2(k=k,threshold=threshold,limits=limits,nms=nms)

    def getBoxesV1(self,k=1000,threshold=0.5,
                 limits=None,
                 nms=None):
        '''
        :param k:
        :param threshold:
        :param nms_threshold:
        :param limits:
        :param nms: parameters is boxes,labels,confidence
        :param classes_wise_nms:
        :param use_soft_nms:
        :return:
        '''
        if nms is None:
            nms = functools.partial(wop.boxes_nms,threshold=0.4,classes_wise=True)
        with tf.variable_scope("GetBoxes"):
            probs = self.score_converter(self.logits)
            self.boxes,self.labels,self.probs,self.indices,self.boxes_lens = \
                od.get_predictionv2(
                    class_prediction=probs,
                    bboxes_regs=self.regs,
                    proposal_bboxes=self.anchors,
                    threshold=threshold,
                    limits=limits,
                    candiate_nr=k,
                    classes_wise=self.pred_bboxes_classwise,
                    nms=nms)
        return self.boxes,self.labels,self.probs,self.boxes_lens

    def getBoxesV2(self,k=1000,threshold=0.5,
                   limits=None,
                   nms=None):
        '''
        :param k:
        :param threshold:
        :param nms_threshold:
        :param limits:
        :param nms: parameters is boxes,labels,confidence
        :param classes_wise_nms:
        :param use_soft_nms:
        :return:
        '''
        if nms is None:
            nms = functools.partial(wop.boxes_nms,threshold=0.4,classes_wise=True)
        with tf.variable_scope("GetBoxes"):
            probs = self.score_converter(self.logits)
            self.boxes,self.labels,self.probs,self.indices,self.boxes_lens = \
                od.get_predictionv5(
                    class_prediction=probs,
                    bboxes_regs=self.regs,
                    proposal_bboxes=self.anchors,
                    threshold=threshold,
                    limits=limits,
                    candiate_nr=k,
                    classes_wise=self.pred_bboxes_classwise,
                    nms=nms)
        return self.boxes,self.labels,self.probs,self.boxes_lens
    
    '''
    process the situation of batch_size greater than one, target boxes number of each imag is specified by k
    '''
    def getBoxesV3(self, k=1000, nms=None,limits=None,
                  adjust_probability=None):
        with tf.device("/cpu:0"):
            with tf.variable_scope("GetBoxesV3"):
                self.raw_probs = self.score_converter(self.logits)
                probs = self.raw_probs
                if adjust_probability is not None:
                    probs = wnnl.probability_adjust(probs=probs,classes=adjust_probability)
            if nms is None:
                nms = functools.partial(wop.boxes_nms_nr2,threshold=0.5,classes_wise=True,k=k)
            self.boxes,self.labels,self.probs,self.indices= \
                od.get_predictionv6(class_prediction=probs,
                                     bboxes_regs=self.regs,
                                     proposal_bboxes=self.anchors,
                                     nms=nms,
                                     limits=limits,
                                     classes_wise=self.pred_bboxes_classwise)
            self.boxes_lens = tf.convert_to_tensor(np.array([k]),dtype=tf.int32)*tf.ones(shape=tf.shape(self.labels)[0],dtype=tf.int32)
            return self.boxes,self.labels,self.probs,self.boxes_lens

