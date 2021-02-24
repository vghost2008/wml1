#coding=utf-8
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from wtfop.wtfop_ops import boxes_encode1,boxes_encode,probability_adjust,wpad,anchor_generator
import object_detection.od_toolkit as od
import object_detection.bboxes as bboxes
import wml_tfutils as wmlt
import object_detection.wlayers as odl
import object_detection.losses as losses
import wtfop.wtfop_ops as wop
import sys
import logging
import wnnlayer as wnnl
import functools
import wsummary
from collections import Iterable
slim = tf.contrib.slim

'''
注: Detectron2中在proposal boxes进入RCN时会将gt boxes加入proposal boxes中，（训练初期RPN产生的proposal boxes)中可能没有正样本
注:RPN时与gt bboxes IOU最大的anchor box会作为正样本，但在RCN中不需要这样做
'''
class FasterRCNN(object):
    __metaclass__ = ABCMeta
    '''
    num_classes: include background
    input shape: the input image shape, [h,w]
    '''
    def __init__(self,num_classes,input_shape,batch_size=1):
        self.target_layer_shape = []
        self.scales=[]
        self.ratios=[]
        self.anchors=None
        self.np_anchors=None
        #[batch_size,H,W,C]
        self.inputs=None
        self.batch_size = batch_size
        '''
        包含背景
        '''
        self.num_classes = num_classes
        #[H,W]
        self.input_shape = input_shape
        '''
        回归损失乘以reg_loss_weight以增加回归的准确率
        '''
        self.reg_loss_weight = 10.
        self.cls_loss_weight = 1.
        self.bbox_regs = None
        #[batch_size,h*w*anchor_size,2]
        self.rpn_logits=None
        #[batch_size,h*w*anchor_size,4]
        self.rpn_regs=None
        self.rpn_gtlabels=None
        self.rpn_gtregs=None
        self.rpn_gtscores=None
        self.proposal_boxes=None
        self.proposal_boxes_prob=None
        #[batch_size,pos_nr+neg_nr]
        self.rcn_gtlabels=None
        self.rcn_gtregs=None
        self.rcn_gtscores=None
        #[batch_size,pos_nr+neg_nr,num_classes]
        self.rcn_logits=None
        #[batch_size,pos_nr+neg_nr,num_classes-1,4]
        self.rcn_regs=None
        #[batch_size,pos_nr+neg_nr], 表示self.rcn_gt***对应的proposal_box索引
        self.rcn_indices = None
        '''
        batch_size,pos_nr+neg_nr]
        the value representation rcn self.rcn_gt*** to the index of ground truth box(in the order of input to encode rcn boxes).
        the background's value is -1
        '''
        self.rcn_anchor_to_gt_indices = None
        self.rcn_bboxes_lens = None
        '''
        the output of rpn is called proposal_boxes, after call encodeRCNBoxes, most of the proposal_boxes are removed, 
        the left boxes is also called proposal_boxes, in order to distinguish these two different proposal boxes set, we 
        now call the later as rcn_proposal_boxes, but the old name is also kept for compatibility.
        [batch_size,pos_nr+neg_nr,4]
        '''
        self.rcn_proposal_boxes=None
        self.finally_boxes=None
        self.finally_boxes_prob=None
        self.finally_boxes_label=None
        self.train_base_net = False
        self.train_rpn = False
        self.train_rcn = False
        self.anchor_remove_indices = None
        self.finally_indices = None
        #[batch_size,h,w,C]
        self.fsbp_net = None
        #[batch_size*(pos_nr+neg_nr),bin_h,bin_w,C]
        self.ssbp_net = None
        #[batch_size*(pos_nr+neg_nr),bin_h,bin_w,C]
        self.net_after_roi = None
        self.rcn_batch_size  = None
        self.rcn_box_nr = None
        #[batch_size*(pos_nr+neg_nr),C]
        self.rcn_raw_probs = None
        self.rpn_scope = "FirstStageBoxPredictor"
        self.rcn_scope = "SecondStageFeatureExtractor"
        '''
        anchor_size用于表示每个地方的anchor_boxes数量，为len(ratios)*len(scales)
        '''
        self.anchor_size=None
        self.pred_bboxes_classwise = False
        self.pred_rcn_bboxes_classwise = True
        self.train_bn = True

    def train_base_net_bn(self):
        return self.train_base_net and self.train_bn

    def train_rpn_bn(self):
        return self.train_rpn and self.train_bn

    def train_rcn_bn(self):
        return self.train_rcn and self.train_bn


    def buildFasterRCNNet(self):
        self.buildRCNNet()

    def buildBaseNet(self,bimage):
        self.base_net = self._buildBaseNet(bimage)
        return self.base_net

    def buildRPNNet(self,bimage):

        self.buildBaseNet(bimage)
        return self.pureBuildRPNNet(self.base_net)

    def pureBuildRPNNet(self,base_net=None,channel=512):
        self.rpnFeatureExtractor(base_net,channel)
        rpn_regs,rpn_logits = self.rpnBoxPredictor()
        regs_shape = rpn_regs.get_shape().as_list()
        logits_shape = rpn_logits.get_shape().as_list()
        if rpn_logits.get_shape().is_fully_defined():
            anchor_nr = logits_shape[1]*logits_shape[2]*logits_shape[3]
        else:
            anchor_nr = tf.reduce_prod(tf.shape(rpn_logits)[1:4])
        self.rpn_regs = wmlt.reshape(rpn_regs,[regs_shape[0],anchor_nr,regs_shape[-1]])
        self.rpn_logits = wmlt.reshape(rpn_logits,[logits_shape[0],anchor_nr,logits_shape[-1]])

        return self.proposal_boxes,self.proposal_boxes_prob

    def _rpnFeatureExtractor(self,base_net=None,channel=512):
        return slim.conv2d(base_net,channel,[3,3],padding="SAME",activation_fn=tf.nn.relu6,normalizer_fn=None)

    def rpnFeatureExtractor(self,base_net=None,channel=512):
        if base_net is None:
            base_net = self.base_net
        self.fsbp_net = self._rpnFeatureExtractor(base_net,channel)

    '''
    base_net: buildBaseNet的返回值,设base_net的shape为bn_shape
    返回:
    [rpn_regs,rpn_logits]
    rpn_regs的shape为[bn_shape[0],bn_shape[1],bn_shape[2],self.anchor_size,4], 用于表示每个位置的所有anchorbox到proposal box的变换参数
    rpn_logits的shape为[bn_shape[0],bn_shape[1],bn_shape[2],self.anchor_size,2], 用于对第个位置的所有anchorbox分类，0为背影，1为目标
    '''
    def rpnBoxPredictor(self):
        with tf.variable_scope("FirstStageBoxPredictor"):
            if self.fsbp_net.get_shape().is_fully_defined():
                base_shape = self.fsbp_net.get_shape().as_list()
            else:
                base_shape = tf.shape(self.fsbp_net)
            regs_net = slim.conv2d(self.fsbp_net,4*self.anchor_size,[1,1],activation_fn=None,normalizer_fn=None,scope="BoxEncodingPredictor")
            regs_net = wmlt.reshape(regs_net,[base_shape[0],base_shape[1],base_shape[2],self.anchor_size,4])
            logits_net = slim.conv2d(self.fsbp_net,2*self.anchor_size,[1,1],activation_fn=None,normalizer_fn=None,scope="ClassPredictor")
            logits_net = wmlt.reshape(logits_net,[base_shape[0],base_shape[1],base_shape[2],self.anchor_size,2])
            return regs_net,logits_net

    @abstractmethod
    def getTargetLayerShape(self):
        #return the shape of rpn input feature map (only h,w)
        pass

    @abstractmethod
    def _buildBaseNet(self,bimage):
        pass

    '''
    proposal_boxes:候选bboxes,[X,4]
    返回:
    rcn_regs,rcn_logits
    rcn_regs的shape为[batch_size,box_nr,num_classes,4]
    rcn_logits的shape为[batch_size,box_nr,num_classes]
    '''
    def rcnFeatureExtractor(self,base_net,proposal_boxes,roipooling,bin_size=7,reuse=False):
        net_channel = base_net.get_shape().as_list()[-1]

        batch_index,batch_size,box_nr = self.rcn_batch_index_helper(proposal_boxes)
        net = roipooling(base_net,proposal_boxes,batch_index,bin_size,bin_size)
        self.net_after_roi = net
        net = wmlt.reshape(net,[batch_size*box_nr,bin_size,bin_size,net_channel])
        self.ssbp_net = self._rcnFeatureExtractor(net,reuse)
        self.rcn_batch_size,self.rcn_box_nr = batch_size,box_nr

    '''
    proposal_boxes:候选bboxes,[X,4]
    返回:
    rcn_regs,rcn_logits
    rcn_regs的shape为[batch_size,box_nr,num_classes,4]
    rcn_logits的shape为[batch_size,box_nr,num_classes]
    '''
    @abstractmethod
    def _rcnFeatureExtractor(self,net,reuse=False):
        pass

    def rcnBoxPredictor(self,batch_size,box_nr,reuse=False):
        with tf.variable_scope("SecondStageBoxPredictor"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            if isinstance(self.ssbp_net,Iterable):
                assert len(self.ssbp_net)==2,"error ssbp_net length"
                ssbp_net_cls,ssbp_net_bbox = self.ssbp_net
                if len(ssbp_net_bbox.shape)==4:
                    ssbp_net_bbox = tf.reduce_mean(ssbp_net_bbox,axis=[1,2],keepdims=False)
                    
                if len(ssbp_net_cls.shape) == 4:
                    ssbp_net_cls = tf.reduce_mean(ssbp_net_cls, axis=[1, 2], keepdims=False)
            else:
                ssbp_net_bbox = tf.reduce_mean(self.ssbp_net,axis=[1,2],keepdims=False)
                ssbp_net_cls = ssbp_net_bbox
            #分类子网络
            rcn_logits = slim.fully_connected(ssbp_net_cls, self.num_classes,
                                              activation_fn = None,
                                              weights_regularizer=None, biases_regularizer=None,
                                              scope="ClassPredictor")
            rcn_logits = tf.reshape(rcn_logits,[batch_size,box_nr,self.num_classes])

            #回归子网络
            if self.pred_rcn_bboxes_classwise:
                pos_num_classes = self.num_classes-1
                rcn_regs = slim.fully_connected(ssbp_net_bbox, 4 * pos_num_classes,
                                                scope="BoxEncodingPredictor",
                                                weights_regularizer=None, biases_regularizer=None,
                                                activation_fn=None)
                rcn_regs = tf.reshape(rcn_regs,[batch_size,box_nr,pos_num_classes,4])
            else:
                rcn_regs = slim.fully_connected(ssbp_net_bbox, 4,
                                                scope="BoxEncodingPredictor",
                                                weights_regularizer=None, biases_regularizer=None,
                                                activation_fn=None)
                rcn_regs = tf.reshape(rcn_regs,[batch_size,box_nr,4])

            return rcn_regs,rcn_logits

    def buildRCNNet(self,roipooling=odl.DFROI(),proposal_boxes=None,bin_size=7,base_net=None,reuse=False):
        if base_net is None:
            base_net = self.base_net
        if proposal_boxes is None:
            proposal_boxes = self.proposal_boxes
        self.rcnFeatureExtractor(base_net,proposal_boxes,roipooling,bin_size,reuse)
        self.rcn_regs,self.rcn_logits = self.rcnBoxPredictor(self.rcn_batch_size,self.rcn_box_nr,reuse)
        if self.pred_rcn_bboxes_classwise:
            assert len(self.rcn_regs.get_shape().as_list())==4,"FasterRCNN: rcn regs should be a 4 dims tensor."
        else:
            assert len(self.rcn_regs.get_shape().as_list())==3,"FasterRCNN: rcn regs should be a 3 dims tensor."
        assert len(self.rcn_logits.get_shape().as_list())==3,"FasterRCNN: rcn logits should be a 3 dims tensor."
        return self.rcn_regs,self.rcn_logits

    #在训练RPN时生成RCN网络参数
    def buildFakeRCNNet(self,roipooling=odl.DFROI()):
        self.proposal_boxes = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,8,4])
        return self.buildRCNNet(roipooling=roipooling)

    '''
    Detectron2中，neg_threshold=0.3, pos_threshold=0.7
    '''
    def encodeRPNBoxes(self,gbboxes, glabels,lens,pos_threshold=0.7,neg_threshold=0.3,max_labels_value=1):
        with tf.name_scope("EncodeRPNBoxes"):
            rpn_gtregs, rpn_gtlabels, rpn_gtscores,remove_indices,_ = boxes_encode(bboxes=self.anchors,
                                                                   gboxes=gbboxes,
                                                                   glabels=glabels,
                                                                   length=lens,
                                                                   pos_threshold=pos_threshold,
                                                                   neg_threshold=neg_threshold)
            if max_labels_value is not None:
                '''
                rpn网络中分类只分背景与目标，类别号为0背景，1目标
                '''
                rpn_gtlabels = tf.clip_by_value(rpn_gtlabels,0,max_labels_value)
            self.rpn_gtregs = rpn_gtregs
            self.rpn_gtlabels = rpn_gtlabels
            self.rpn_gtscores = rpn_gtscores
            self.anchor_remove_indices = remove_indices
            return rpn_gtregs, rpn_gtlabels, rpn_gtscores,remove_indices
    '''
    ***** 执行后self.proposal_boxes会变成pos_threashold+neg_threshold个*****
    注：Faster-RCNN原文中pos_threshold=0.5, neg_threshold=0.5, Detectron2中也是使用的这一设定
    需要在buildRCNNet前调用
    gbboxes:[batch_size,X,4]
    glabels:[batch_size,X]
    lens:[batch_size]
    neg_nr: number of negative box in every example
    pos_nr: number of postive box in every example
    '''
    def encodeRCNBoxes(self, gbboxes, glabels,lens,
                       pos_threshold=0.5, neg_threshold=0.5,
                       proposal_boxes=None,neg_nr=64,pos_nr=64):
        with tf.name_scope("EncodeRCNBoxes"):
            if proposal_boxes is None:
                proposal_boxes = self.proposal_boxes
            rcn_gtregs, rcn_gtlabels, rcn_gtscores,remove_indices,indices = boxes_encode(bboxes=proposal_boxes,
                                                                                  gboxes=gbboxes,
                                                                                  glabels=glabels,
                                                                                  length=lens,
                                                                                  pos_threshold=pos_threshold,
                                                                                  neg_threshold=neg_threshold,
                                                                                  max_overlap_as_pos=False)
            keep_indices = tf.stop_gradient(tf.logical_not(remove_indices))
            self.rcn_gtlabels,self.rcn_indices = \
                tf.map_fn(lambda x:self.selectRCNBoxes(x[0],x[1],neg_nr=neg_nr,pos_nr=pos_nr),
                          elems = (rcn_gtlabels,keep_indices),
                          dtype=(tf.int32,tf.int32),
                          back_prop=False,
                          parallel_iterations=self.batch_size)
            self.rcn_anchor_to_gt_indices = tf.stop_gradient(wmlt.batch_gather(indices,self.rcn_indices))
            self.rcn_proposal_boxes = tf.stop_gradient(wmlt.batch_gather(proposal_boxes,self.rcn_indices))
            self.proposal_boxes = self.rcn_proposal_boxes #for compatibility
            self.rcn_gtregs = wmlt.batch_gather(rcn_gtregs,self.rcn_indices)
            self.rcn_gtscores = wmlt.batch_gather(rcn_gtscores,self.rcn_indices)

            return self.rcn_gtregs, self.rcn_gtlabels, self.rcn_gtscores
    '''
    随机在输入中选择指定数量的box
    要求必须结果包含neg_nr+pos_nr个结果，如果总数小于这么大优先使用背影替代，然后使用随机替代
    注:Fast-RCNN原文中随机取25% IOU>0.5的目标, 75%IOU<=0.5的目标, 一个批次总共取128个目标，也就是每张图取128/batch_size个目标
    注:Detectron2中每个图像都会取512个目标，远远大于Faster-RCNN原文的数量, 比例与原文一样(25%的正样本)
    labels:[X]
    keep_indices:[X],tf.bool
    返回:
    r_labels:[nr]
    r_indices:[nr]
    '''
    @staticmethod
    def selectRCNBoxes(labels, keep_indices, neg_nr, pos_nr):
        nr = neg_nr + pos_nr
        r_indices = tf.range(0, tf.shape(labels)[0])
        labels = tf.boolean_mask(labels, keep_indices)
        r_indices = tf.boolean_mask(r_indices, keep_indices)
        total_neg_nr = tf.reduce_sum(tf.cast(tf.equal(labels, 0), tf.int32))
        total_pos_nr = tf.shape(labels)[0] - total_neg_nr

        with tf.name_scope("SelectRCNBoxes"):
            def random_select(labels, indices, size):
                with tf.name_scope("random_select"):
                    data_nr = tf.shape(labels)[0]
                    indexs = tf.range(data_nr)
                    indexs = wpad(indexs, [0, size - data_nr])
                    indexs = tf.random_shuffle(indexs)
                    indexs = tf.random_crop(indexs, [size])
                    labels = tf.gather(labels, indexs)
                    indices = tf.gather(indices, indexs)
                    return labels, indices

            # positive and negitave number satisfy the requement
            def selectRCNBoxesM0():
                with tf.name_scope("M0"):
                    n_mask = tf.equal(labels, 0)
                    p_mask = tf.logical_not(n_mask)
                    n_labels = tf.boolean_mask(labels, n_mask)
                    n_indices = tf.boolean_mask(r_indices, n_mask)
                    p_labels = tf.boolean_mask(labels, p_mask)
                    p_indices = tf.boolean_mask(r_indices, p_mask)
                    p_labels, p_indices = random_select(p_labels, p_indices, pos_nr)
                    n_labels, n_indices = random_select(n_labels, n_indices, neg_nr)

                    return tf.concat([p_labels, n_labels], axis=0), tf.concat([p_indices, n_indices], axis=0)

            # process the default situation
            def selectRCNBoxesM1():
                with tf.name_scope("M1"):
                    return random_select(labels, r_indices, nr)

            # positive dosen't satisfy the requement
            def selectRCNBoxesM2():
                with tf.name_scope("M2"):
                    n_mask = tf.equal(labels, 0)
                    p_mask = tf.logical_not(n_mask)
                    n_labels = tf.boolean_mask(labels, n_mask)
                    n_indices = tf.boolean_mask(r_indices, n_mask)
                    p_labels = tf.boolean_mask(labels, p_mask)
                    p_indices = tf.boolean_mask(r_indices, p_mask)
                    with tf.name_scope("Select"):
                        n_labels, n_indices = \
                            random_select(n_labels, n_indices, nr - total_pos_nr)
                    return tf.concat([p_labels, n_labels], axis=0), tf.concat([p_indices, n_indices], axis=0)

            # positive and negative is empty
            def selectRCNBoxesM3():
                with tf.name_scope("M3"):
                    # boxes = tf.constant([[0.0,0.0,0.001,0.001]],dtype=tf.float32)*tf.ones([nr,4],dtype=tf.float32)
                    # boxes_regs = tf.zeros_like(boxes,dtype=tf.float32)
                    labels = tf.constant([0]) * tf.ones([nr], dtype=tf.int32)
                    # scores = tf.ones_like(labels,dtype=tf.float32)
                    return labels, tf.zeros_like(labels)

            r_labels, r_indices = tf.case({
                tf.logical_and(total_pos_nr >= pos_nr, total_neg_nr >= neg_nr): selectRCNBoxesM0,
                tf.logical_and(tf.logical_and(total_pos_nr < pos_nr, total_neg_nr >= neg_nr),
                               total_pos_nr > 0): selectRCNBoxesM2,
                tf.equal(tf.shape(labels)[0], 0): selectRCNBoxesM3
            },
                default=selectRCNBoxesM1,
                exclusive=True)
            r_labels.set_shape([nr])
            r_indices.set_shape([nr])
            return r_labels, r_indices

    def evalRPNNet(self, gbboxes,proposal_boxes,proposal_boxes_prob,pos_threshold=0.7):
        with tf.name_scope("EncodeRCNBoxes"):
            if proposal_boxes.get_shape().ndims == 3:
                proposal_boxes = tf.squeeze(proposal_boxes,axis=0)
            if proposal_boxes_prob.get_shape().ndims == 2:
                proposal_boxes_prob = tf.squeeze(proposal_boxes_prob,axis=0)
            assert proposal_boxes.get_shape().ndims==2, "Proposal boxes's dims should be 2."
            assert proposal_boxes_prob.get_shape().ndims==1, "Proposal boxes_prob's dims should be 1."

            glabels = tf.ones(shape=[tf.shape(proposal_boxes)[0]],dtype=tf.int32)
            rcn_gtregs, rcn_gtlabels, rcn_gtscores,remove_indices =\
                boxes_encode1(bboxes=proposal_boxes,
                               gboxes=gbboxes,
                               glabels=glabels,
                               pos_threshold=pos_threshold,
                               neg_threshold=0.0)
            if remove_indices is not None:
                if remove_indices.get_shape().ndims > 1:
                    remove_indices = tf.squeeze(remove_indices,axis=0)
                remove_indices = tf.logical_or(remove_indices,tf.equal(rcn_gtlabels,0))
            else:
                remove_indices = tf.equal(rcn_gtlabels,0)
            keep_indices = tf.logical_not(remove_indices)
            proposal_boxes = tf.boolean_mask(proposal_boxes,keep_indices)
            proposal_boxes_prob = tf.boolean_mask(proposal_boxes_prob,keep_indices)
            return proposal_boxes,proposal_boxes_prob
    '''
    反回的为每一层，每一个位置，每一个大小，每一个比率的anchorbox
    shape为[-1,4],最后一维为[ymin,xmin,ymax,xmax](相对坐标)
    '''
    def getAnchorBoxes(self):
        shape = self.getTargetLayerShape()
        anchors = bboxes.get_anchor_bboxes(shape,sizes=self.scales,ratios=self.ratios)
        self.np_anchors = anchors
        anchors = tf.convert_to_tensor(anchors)
        anchors = tf.expand_dims(anchors,axis=0)
        self.anchors = anchors*tf.ones([self.batch_size]+anchors.get_shape().as_list()[1:],dtype=tf.float32)

        return self.anchors

    def getAnchorBoxesV2(self):
        shape = self.getTargetLayerShape()
        anchors = bboxes.get_anchor_bboxesv2(shape,sizes=self.scales,ratios=self.ratios)
        self.np_anchors = anchors
        anchors = tf.convert_to_tensor(anchors)
        anchors = tf.expand_dims(anchors,axis=0)
        self.anchors = anchors*tf.ones([self.batch_size]+anchors.get_shape().as_list()[1:],dtype=tf.float32)
        return self.anchors

    '''
    反回的为每一层，每一个位置，每一个比率,每一个大小的anchorbox
    shape为[-1,4],最后一维为[ymin,xmin,ymax,xmax](相对坐标)
    '''
    def getAnchorBoxesV3(self):
        shape = self.getTargetLayerShape()
        anchors = bboxes.get_anchor_bboxesv3(shape,sizes=self.scales,ratios=self.ratios,is_area=False)
        self.np_anchors = anchors
        anchors = tf.convert_to_tensor(anchors)
        anchors = tf.expand_dims(anchors,axis=0)
        self.anchors = anchors*tf.ones([self.batch_size]+anchors.get_shape().as_list()[1:],dtype=tf.float32)

        return self.anchors

    def getAnchorBoxesV4(self,shape,img_size):
        anchors = anchor_generator(shape=shape,size=img_size,scales=self.scales,aspect_ratios=self.ratios)
        anchors = tf.expand_dims(anchors,axis=0)
        self.anchors = anchors*tf.ones([self.batch_size]+[1]*(anchors.get_shape().ndims-1),dtype=tf.float32)
        return self.anchors

    def getAnchorBoxesByBaseNetAndImgV4(self,img,base_net):
        if img.get_shape().is_fully_defined():
            img_size = img.get_shape().as_list()[1:3]
        else:
            img_size = tf.shape(img)[1:3]
        if base_net.get_shape().is_fully_defined():
            shape = base_net.get_shape().as_list()[1:3]
        else:
            shape = tf.shape(base_net)[1:3]
        return self.getAnchorBoxesV4(shape,img_size)
    '''
    用于计算RPN网络的损失
    '''
    def getRPNLoss(self,loss=None,scope="RPNLoss"):
        if loss is None:
            '''
            Detectron2中，RPN在每个图像中随机取256个样本，其中正样本50%(如果正样本没取够用负样本填充)
            '''
            loss = losses.ODLoss(num_classes=2,
                                 reg_loss_weight=self.reg_loss_weight,
                                 cls_loss_weight=self.cls_loss_weight,
                                 scope=scope,
                                 classes_wise=False,do_sample=True,
                                 sample_type=losses.ODLoss.SAMPLE_TYPE_UNIFORM,
                                 sample_size=256)
        return loss(gregs=self.rpn_gtregs,
                   glabels=self.rpn_gtlabels,
                   classes_logits=self.rpn_logits,
                   bboxes_regs=self.rpn_regs,
                   bboxes_remove_indices=self.anchor_remove_indices)
    '''
    用于计算RCN网络的损失
    '''
    def getRCNLoss(self,labels=None,loss=None,scores=None,call_back=None,scope="RCNLoss"):
        if labels is None:
            labels = self.rcn_gtlabels
        if scores is not None:
            if loss is None:
                loss = losses.ODLoss(num_classes=self.num_classes,
                                     reg_loss_weight=self.reg_loss_weight,
                                     cls_loss_weight=self.cls_loss_weight,
                                     scope=scope,classes_wise=self.pred_rcn_bboxes_classwise,do_sample=False)
            return loss(gregs=self.rcn_gtregs,
                       glabels=labels,
                       scores=scores,
                       classes_logits=self.rcn_logits,
                       bboxes_regs=self.rcn_regs)
        else:
            if loss is None:
                loss = losses.ODLoss(num_classes=self.num_classes,
                                     reg_loss_weight=self.reg_loss_weight,
                                     cls_loss_weight=self.cls_loss_weight,
                                     scope=scope,classes_wise=self.pred_rcn_bboxes_classwise)
            return loss(gregs=self.rcn_gtregs,
                       glabels=labels,
                       classes_logits=self.rcn_logits,
                       bboxes_regs=self.rcn_regs,call_back=call_back)

    '''
    Default use getProposalBoxesV4, which use the same strategy as Detectron2
    '''
    def getProposalBoxes(self,k=1000,nms_threshold=0.7):
        return self.getProposalBoxesV4(k=k,nms_threshold=nms_threshold)

    def getProposalBoxesV0(self,k=1000,threshold=0.5,nms_threshold=0.1):
        with tf.variable_scope("RPNProposalBoxes"):
            self.proposal_boxes,labels,self.proposal_boxes_prob,_ =\
                od.get_prediction(class_prediction=tf.nn.softmax(self.rpn_logits),
                                     bboxes_regs=self.rpn_regs,
                                     proposal_bboxes=self.anchors,
                                     threshold=threshold,
                                     nms_threshold=nms_threshold,
                                     candiate_nr=k,
                                     classes_wise=False)
            self.proposal_boxes = bboxes.tf_correct_yxminmax_boxes(self.proposal_boxes)
        return self.proposal_boxes,self.proposal_boxes_prob
    '''
    get exactly k proposal boxes, the excess boxes was remove by nms, the nms's threshold is dynamic changed to fit the requirement boxes size.
    befory nms, the boxes wes sorted by thre probability.
    k: output boxes number
    '''
    def getProposalBoxesV2(self,k=1000,candiate_multipler=10):
        with tf.variable_scope("RPNProposalBoxes"):
            self.proposal_boxes,labels,self.proposal_boxes_prob =\
                od.get_proposal_boxes(class_prediction=tf.nn.softmax(self.rpn_logits),
                                     bboxes_regs=self.rpn_regs,
                                     proposal_bboxes=self.anchors,
                                     candiate_nr=k,
                                     candiate_multipler=candiate_multipler,
                                     classes_wise=False)
            self.proposal_boxes = bboxes.tf_correct_yxminmax_boxes(self.proposal_boxes)
        return self.proposal_boxes,self.proposal_boxes_prob

    '''
    get exactly k proposal boxes, the excess boxes was remove by probability.
    k: output boxes number
    output:
    [batch_size,k,4],[batch_size,k]
    '''
    def getProposalBoxesV3(self,k=1000):
        with tf.variable_scope("RPNProposalBoxes"):
            self.proposal_boxes,labels,self.proposal_boxes_prob =\
                od.get_proposal_boxesv2(class_prediction=tf.nn.softmax(self.rpn_logits),
                                     bboxes_regs=self.rpn_regs,
                                     proposal_bboxes=self.anchors,
                                     candiate_nr=k,
                                     classes_wise=False)
            self.proposal_boxes = bboxes.tf_correct_yxminmax_boxes(self.proposal_boxes)
        return self.proposal_boxes,self.proposal_boxes_prob

    '''
    Detectron2在实现时先从每个图选12000个得分最高的box再使用threshold=0.7的nms处理，再取前2000个，数量都多于Faster-RCNN原文的1000个
    在预测时数量则减为6000与1000
    This implement is compitable with Detectron2
    get exactly k proposal boxes, the excess boxes was remove by heristic method.
    k: output boxes number
    nms_threshold: NMS operation threshold.
    output:proposal_boxes:[batch_size,k,4]
    output:proposal_prob:[batch_size,k]
    '''
    def getProposalBoxesV4(self,k=1000,nms_threshold=0.7):
        with tf.variable_scope("RPNProposalBoxes"):
            self.proposal_boxes,labels,self.proposal_boxes_prob = \
                od.get_proposal_boxesv3(class_prediction=self.rpn_logits,
                                     bboxes_regs=self.rpn_regs,
                                     proposal_bboxes=self.anchors,
                                     candiate_nr=k,
                                     nms_threshold=nms_threshold,
                                     classes_wise=False)
            self.proposal_boxes = bboxes.tf_correct_yxminmax_boxes(self.proposal_boxes)
        return self.proposal_boxes,self.proposal_boxes_prob

    '''
    only process the situation of batch_size equals one
    '''
    def getBoxes(self,k=1000,threshold=0.5,nms_threshold=0.1,
                 proposal_boxes=None,limits=None,
                 adjust_probability=None,classes_wise_nms=True):
        if proposal_boxes is None:
            proposal_boxes = self.proposal_boxes
        with tf.device("/cpu:0"):
            with tf.variable_scope("RCNGetBoxes"):
                self.rcn_raw_probs = tf.nn.softmax(self.rcn_logits)
                probs = self.rcn_raw_probs
                if adjust_probability is not None:
                    probs = tf.squeeze(probs,axis=0)
                    probs = probability_adjust(probs=probs,classes=adjust_probability)
                    probs = tf.expand_dims(probs,axis=0)
                self.finally_boxes,self.finally_boxes_label,self.finally_boxes_prob,\
                    self.finally_indices = od.get_prediction(
                                         class_prediction=probs,
                                         bboxes_regs=self.rcn_regs,
                                         proposal_bboxes=proposal_boxes,
                                         threshold=threshold,
                                         nms_threshold=nms_threshold,
                                         limits=limits,
                                         candiate_nr=k,
                                         classes_wise=True,
                                         classes_wise_nms=classes_wise_nms)
        return self.finally_boxes,self.finally_boxes_label,self.finally_boxes_prob,tf.shape(self.finally_boxes_label)

    '''
    process the situation of batch_size greater than one, target boxes number of each imag is very different, so the 
    boxes number of each image is return by lens
    nms:是否使用softnms,使用soft nms与不使用soft nms时, nms_threshold的意义有很大的区别， 不使用soft nms时，nms_threshold表示
IOU小于nms_threshold的两个bbox为不同目标，使用soft nms时，nms_threshold表示得分高于nms_threshold的才是真目标
    '''
    def getBoxesV2(self,k=1000,threshold=0.5,proposal_boxes=None,limits=None,
                   adjust_probability=None,
                   nms=None):
        if nms is None:
            nms = functools.partial(wop.boxes_nms,threshold=0.4,classes_wise=True)
        if proposal_boxes is None:
            proposal_boxes = self.proposal_boxes
        with tf.device("/cpu:0"):
            with tf.variable_scope("RCNGetBoxes"):
                self.rcn_raw_probs = tf.nn.softmax(self.rcn_logits)
                probs = self.rcn_raw_probs
                if adjust_probability is not None:
                    probs = tf.squeeze(probs,axis=0)
                    probs = probability_adjust(probs=probs,classes=adjust_probability)
                    probs = tf.expand_dims(probs,axis=0)
                self.finally_boxes,self.finally_boxes_label,self.finally_boxes_prob,self.finally_indices,\
                self.rcn_bboxes_lens = od.get_predictionv2(
                                         class_prediction=probs,
                                         bboxes_regs=self.rcn_regs,
                                         proposal_bboxes=proposal_boxes,
                                         threshold=threshold,
                                         limits=limits,
                                         candiate_nr=k,
                                         classes_wise=self.pred_rcn_bboxes_classwise,
                                         nms=nms)
        return self.finally_boxes,self.finally_boxes_label,self.finally_boxes_prob,self.rcn_bboxes_lens

    '''
    process the situation of batch_size greater than one, target boxes number of each imag is specified by k
    '''
    def getBoxesV3(self,k=1000,proposal_boxes=None,limits=None,
                   adjust_probability=None,nms=None):
        if proposal_boxes is None:
            proposal_boxes = self.proposal_boxes
        if k is None and nms is None:
            print("Error arguments of getBoxesV3, k and nms can't be None at the same time.")
        with tf.device("/cpu:0"):
            with tf.variable_scope("RCNGetBoxes"):
                self.rcn_raw_probs = tf.nn.softmax(self.rcn_logits)
                probs = self.rcn_raw_probs
                if adjust_probability is not None:
                    probs = wnnl.probability_adjust(probs=probs,classes=adjust_probability)
                if nms is None:
                    nms = functools.partial(wop.boxes_nms_nr2, classes_wise=True, threshold=0.4, k=k)
                self.finally_boxes,self.finally_boxes_label,self.finally_boxes_prob,self.finally_indices =\
                od.get_predictionv3(class_prediction=probs,
                                         bboxes_regs=self.rcn_regs,
                                         proposal_bboxes=proposal_boxes,
                                         limits=limits,
                                         classes_wise=True,
                                         nms=nms
                                         )
        self.rcn_bboxes_lens = tf.ones(tf.shape(self.finally_boxes_label)[0],dtype=tf.int32)*tf.shape(self.finally_boxes_label)[1]
        return self.finally_boxes,self.finally_boxes_label,self.finally_boxes_prob
    '''
    对所有的boxes解码，并且最终boxes的顺序不改变，数量不变
    '''
    def getBoxesV4(self,no_background=True):
        proposal_boxes = self.proposal_boxes
        with tf.device("/cpu:0"):
            with tf.variable_scope("RCNGetBoxes"):
                self.rcn_raw_probs = tf.nn.softmax(self.rcn_logits)
                probs = self.rcn_raw_probs
                bboxes,labels,probs = \
                    od.get_predictionv7(class_prediction=probs,
                                        bboxes_regs=self.rcn_regs,
                                        proposal_bboxes=proposal_boxes,
                                        classes_wise=self.pred_rcn_bboxes_classwise,
                                        no_background=no_background)
                return bboxes,labels,probs

    def getRCNBBoxesPredictQuality(self,gbboxes,glabels,glens):
        proposal_boxes = self.proposal_boxes
        with tf.device("/cpu:0"):
            with tf.variable_scope("RCNGetBoxes"):
                self.rcn_raw_probs = tf.nn.softmax(self.rcn_logits)
                probs = self.rcn_raw_probs
                bboxes,labels,probs = \
                    od.get_predictionv7(class_prediction=probs,
                                        bboxes_regs=self.rcn_regs,
                                        proposal_bboxes=proposal_boxes,
                                        classes_wise=True)
                def fn(sbboxes,slabels,sgbboxes,sglabels,sglen):
                    mask = tf.greater(slabels,0)
                    partation_data = tf.dynamic_partition(sbboxes,tf.cast(mask,tf.int32),2)
                    regs, labels, scores,remove_indices = boxes_encode1(bboxes=partation_data[1],
                                                                                   gboxes=sgbboxes[:sglen],
                                                                                   glabels=sglabels[:sglen],
                                                                                   pos_threshold=0.0,
                                                                                   neg_threshold=0.0)
                    indices = tf.dynamic_partition(tf.range(tf.shape(slabels)[0]),tf.cast(mask,tf.int32),2)
                    scores0 = tf.ones_like(indices[0],dtype=tf.float32)

                    return tf.dynamic_stitch(indices,[scores0,scores])
                return tf.map_fn(lambda x:fn(x[0],x[1],x[2],x[3],x[4]),elems=(bboxes,labels,gbboxes,glabels,glens),dtype=tf.float32)


    def getRCNProbibality(self,adjust_probability=None):
        with tf.device(":/cpu:0"):
            class_prediction = tf.nn.softmax(self.rcn_logits)
            if adjust_probability is not None:
                class_prediction = tf.squeeze(class_prediction, axis=0)
                class_prediction = probability_adjust(probs=class_prediction,classes=adjust_probability)
                class_prediction = tf.expand_dims(class_prediction, axis=0)
            class_prediction = class_prediction[:, :, 1:]
            probability, labels = tf.nn.top_k(class_prediction, k=1)
            labels = tf.reshape(labels,[1,-1])+1
            probability = tf.reshape(probability,[1,-1])
        return labels,probability

    def setRPNBatchInputs(self,bimage, brpn_gtregs, brpn_gtlabels, brpn_gtscores,brpn_remove_indices):
        self.inputs = bimage
        self.rpn_gtregs = brpn_gtregs
        self.rpn_gtlabels = brpn_gtlabels
        self.rpn_gtscores = brpn_gtscores
        self.anchor_remove_indices = brpn_remove_indices


    def setRPNBatchInputsForPredict(self,bimage):
        self.inputs = bimage

    def rcn_batch_index_helper(self,proposal_boxes):
        '''
        get the batch index for roi_pooling
        :param proposal_boxes: [batch_size,X,4], the result of RPN network
        :return:
        batch_index:[batch_size,X]
        batch_size: if batch_size if clearly defined, return the batch_size else return -1
        box_nr: if box_nr if clearly defined, return the box_nr(X), else return -1, in this situation the batch_size 
        must be one.
        '''
        if proposal_boxes.get_shape().is_fully_defined():
            pb_shape = proposal_boxes.get_shape().as_list()
            batch_size = pb_shape[0]
            box_nr = pb_shape[1]
            batch_index = tf.expand_dims(tf.range(batch_size), axis=1) * tf.ones([1, pb_shape[1]], dtype=tf.int32)
            assert batch_size == self.batch_size, "network batch size {} is not equal the seted batch size {}.".format(
                batch_size, self.batch_size)
        else:
            shape = proposal_boxes.get_shape().as_list()
    
            if shape[0] is None:
                batch_index = tf.reshape(tf.range(tf.shape(proposal_boxes)[0], dtype=tf.int32),
                                         [tf.shape(proposal_boxes)[0], 1])
                batch_index *= tf.ones(tf.shape(proposal_boxes)[:2], dtype=tf.int32)
                box_nr = shape[1]
                batch_size = -1
                logging.warning(
                    "The network's feature map's batch_size is not fully defined, in this situation, the box_nr must be fully defined.")
            else:
                batch_index = tf.zeros([1, tf.shape(proposal_boxes)[1]], dtype=tf.int32)
                box_nr = -1
                batch_size = 1
                logging.warning(
                    "The network's feature map's shape is not fully defined, in this situation, the batch size must be one.")
                assert batch_size == self.batch_size, "network batch size {} is not equal seted batch size {}.".format(
                    batch_size, self.batch_size)

        return batch_index,batch_size,box_nr

    def get_5d_ssbp_net(self):
        shape = self.ssbp_net.get_shape().as_list()[1:]
        return wmlt.reshape(self.ssbp_net,[self.rcn_batch_size,self.rcn_box_nr]+shape)

    def to_4d_ssbp_net(self,net):
        shape = net.get_shape().as_list()[2:]
        return wmlt.reshape(net,[-1]+shape)

    def lop_rpn_anchors_loss_call_back(self,pmask, nmask,data_labels=None):
        raw_pmask = pmask
        if data_labels is None:
            data_labels = self.rpn_gtlabels
        ###
        pmask = raw_pmask[0]
        labels = data_labels[0]
        boxes = self.anchors[0]
        labels = tf.expand_dims(tf.boolean_mask(labels, pmask), axis=0)
        boxes = tf.expand_dims(tf.boolean_mask(boxes, pmask), axis=0)
        wsummary.detection_image_summary(self.input_img[:1], boxes=boxes, classes=labels, name="RPN_POS_ANCHORS")
        ###
        nmask = nmask[0]
        labels = data_labels[0]
        boxes = self.anchors[0]
        labels = tf.expand_dims(tf.boolean_mask(labels, nmask), axis=0)
        boxes = tf.expand_dims(tf.boolean_mask(boxes, nmask), axis=0)
        wsummary.detection_image_summary(self.input_img[:1], boxes=boxes, classes=labels, name="RPN_NEG_ANCHORS")
        ###

    '''
    At the begining of training, most of the output bboxes (proposal_boxes) of rpn is negative, so there is no positive
    bboxes for rcn, in order to process this situation, we add the distored gtboxes to proposal_boxes, so this function 
    should be called between getProposalBoxes and encodeRCNBoxes
    Note that: Detectron2 also use the same strategy, but they juse simple add the gtboxes to the proposal_boxes
    '''
    def add_ground_truth_boxes_to_proposal_boxes(self,gtboxes,boxes_lens,nr=16,limits=[0.1,0.1,0.1,0.1]):
        with tf.name_scope("add_ground_truth_boxes_to_proposal_boxes"):
            boxes = od.random_select_boxes(gtboxes,boxes_lens,nr)
            if limits is not None:
                boxes = bboxes.random_distored_boxes(boxes,limits=limits,size=1,keep_org=True)
            self.proposal_boxes = tf.concat([self.proposal_boxes,boxes],axis=1)
        return self.proposal_boxes

