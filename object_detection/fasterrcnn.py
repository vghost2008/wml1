#coding=utf-8
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from wtfop.wtfop_ops import boxes_encode1,boxes_encode,probability_adjust,wpad
import object_detection.od_toolkit as od
import object_detection.bboxes as bboxes
import wml_tfutils as wmlt
import object_detection.wlayers as odl
import object_detection.losses as losses
import sys
import logging
import wnnlayer as wnnl


class FasterRCN(object):
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
        self.inputs=None
        self.batch_size = batch_size
        '''
        包含背景
        '''
        self.num_classes = num_classes
        self.input_shape = input_shape
        '''
        回归损失乘以reg_loss_weight以增加回归的准确率
        '''
        self.reg_loss_weight = 3.
        self.bbox_regs = None
        self.rpn_logits=None
        self.rpn_regs=None
        self.rpn_gtlabels=None
        self.rpn_gtregs=None
        self.rpn_gtscores=None
        self.proposal_boxes=None
        self.proposal_boxes_prob=None
        self.rcn_gtlabels=None
        self.rcn_gtregs=None
        self.rcn_gtscores=None
        self.rcn_logits=None
        self.rcn_regs=None
        self.rcn_bboxes_lens = None
        self.finally_boxes=None
        self.finally_boxes_prob=None
        self.finally_boxes_label=None
        self.train_rpn = False
        self.train_rcn = False
        self.anchor_remove_indices = None
        self.finally_indices = None
        self.rpn_scope = "RPN"
        self.rcn_scope = "RCN"
        '''
        anchor_size用于表示每个地方的anchor_boxes数量，为len(ratios)*len(scales)
        '''
        self.anchor_size=None

    def buildFasterRCNNet(self):
        self.buildRCNNet()

    def buildBaseNet(self,bimage):
        self.base_net = self._buildBaseNet(bimage)
        return self.base_net

    def buildRPNNet(self,bimage):

        self.buildBaseNet(bimage)
        return self.pureBuildRPNNet(self.base_net)

    def pureBuildRPNNet(self,base_net):
        rpn_regs,rpn_logits = self._buildRPNNet(base_net)
        with tf.variable_scope(self.rpn_scope):
            regs_shape = rpn_regs.get_shape().as_list()
            logits_shape = rpn_logits.get_shape().as_list()
            anchor_nr = logits_shape[1]*logits_shape[2]*logits_shape[3]
            self.rpn_regs = wmlt.reshape(rpn_regs,[regs_shape[0],anchor_nr,regs_shape[-1]])
            self.rpn_logits = wmlt.reshape(rpn_logits,[logits_shape[0],anchor_nr,logits_shape[-1]])

        return self.proposal_boxes,self.proposal_boxes_prob

    @abstractmethod
    def getTargetLayerShape(self):
        #return the shape of rpn input feature map (only h,w)
        pass

    @abstractmethod
    def _buildBaseNet(self,bimage):
        pass

    '''
    base_net: buildBaseNet的返回值,设base_net的shape为bn_shape
    返回:
    [rpn_regs,rpn_logits]
    rpn_regs的shape为[bn_shape[0],bn_shape[1],bn_shape[2],self.anchor_size,num_classes,4], 用于表示每个位置的所有anchorbox到proposal box的变换参数
    rpn_logits的shape为[bn_shape[0],bn_shape[1],bn_shape[2],self.anchor_size,2], 用于对第个位置的所有anchorbox分类，0为背影，1为目标
    '''
    @abstractmethod
    def _buildRPNNet(self,base_net):
        pass

    '''
    proposal_boxes:候选bboxes,[X,4]
    返回:
    rcn_regs,rcn_logits
    rcn_regs的shape为[batch_size,box_nr,num_classes,4]
    rcn_logits的shape为[batch_size,box_nr,num_classes]
    '''
    @abstractmethod
    def _buildRCNNet(self,base_net,proposal_boxes,roipooling,reuse):
        pass

    def buildRCNNet(self,roipooling=odl.DFROI(),proposal_boxes=None,base_net=None,reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        if proposal_boxes is None:
            proposal_boxes = self.proposal_boxes
        if base_net is None:
            base_net = self.base_net
        self.rcn_regs,self.rcn_logits = self._buildRCNNet(base_net,proposal_boxes,roipooling=roipooling)
        return self.rcn_regs,self.rcn_logits

    #在训练RPN时生成RCN网络参数
    def buildFakeRCNNet(self,roipooling=odl.DFROI()):
        self.proposal_boxes = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,9,4])
        return self.buildRCNNet(roipooling=roipooling)

    def encodeRPNBoxes(self,gbboxes, glabels,lens,pos_threshold=0.7,neg_threshold=0.3):
        with tf.name_scope("EncodeRPNBoxes"):
            rpn_gtregs, rpn_gtlabels, rpn_gtscores,remove_indices = boxes_encode(bboxes=self.anchors,
                                                                   gboxes=gbboxes,
                                                                   glabels=glabels,
                                                                   length=lens,
                                                                   pos_threshold=pos_threshold,
                                                                   neg_threshold=neg_threshold)
            '''
            rpn网络中分类只分背景与目标，类别号为0背景，1目标
            '''
            rpn_gtlabels = tf.clip_by_value(rpn_gtlabels,0,1)
            self.rpn_gtregs = rpn_gtregs
            self.rpn_gtlabels = rpn_gtlabels
            self.rpn_gtscores = rpn_gtscores
            self.anchor_remove_indices = remove_indices
            return rpn_gtregs, rpn_gtlabels, rpn_gtscores,remove_indices
    '''
    neg_nr: number of negative box in every example
    pos_nr: number of postive box in every example
    '''
    def encodeRCNBoxes(self, gbboxes, glabels,lens,
                       pos_threshold=0.7, neg_threshold=0.3,
                       proposal_boxes=None,neg_nr=200,pos_nr=100):
        with tf.name_scope("EncodeRCNBoxes"):
            if proposal_boxes is None:
                proposal_boxes = self.proposal_boxes
            rcn_gtregs, rcn_gtlabels, rcn_gtscores,remove_indices = boxes_encode(bboxes=proposal_boxes,
                                                                                  gboxes=gbboxes,
                                                                                  glabels=glabels,
                                                                                 length=lens,
                                                                                  pos_threshold=pos_threshold,
                                                                                  neg_threshold=neg_threshold)
            keep_indices = tf.logical_not(remove_indices)
            self.proposal_boxes,self.rcn_gtregs,self.rcn_gtlabels,self.rcn_gtscores = \
                tf.map_fn(lambda x:self.selectRCNBoxes(x[0],x[1],x[2],x[3],x[4],neg_nr=neg_nr,pos_nr=pos_nr),
                          elems = (self.proposal_boxes,rcn_gtregs,rcn_gtlabels,rcn_gtscores,keep_indices),
                          dtype=(tf.float32,tf.float32,tf.int32,tf.float32),
                          back_prop=False,
                          parallel_iterations=self.batch_size)

            return self.rcn_gtregs, self.rcn_gtlabels, self.rcn_gtscores
    '''
    随机在输入中选择指定数量的box
    要求必须结果包含neg_nr+pos_nr个结果，如果总数小于这么大优先使用背影替代，然后使用随机替代
    boxes:[X,4]
    labels:[X]
    scores:[X]
    remove_indices:[X]
    返回:
    M=neg_nr+pos_nr
    r_boxes:[M,4]
    r_labels:[M]
    r_scores:[M]
    '''
    @staticmethod
    def selectRCNBoxes(boxes,boxes_regs,labels,scores,keep_indices,neg_nr,pos_nr):
        nr = neg_nr+pos_nr
        boxes = tf.boolean_mask(boxes,keep_indices)
        boxes_regs = tf.boolean_mask(boxes_regs,keep_indices)
        labels = tf.boolean_mask(labels,keep_indices)
        scores = tf.boolean_mask(scores,keep_indices)
        total_neg_nr = tf.reduce_sum(tf.cast(tf.equal(labels, 0), tf.int32))
        total_pos_nr = tf.shape(labels)[0] - total_neg_nr
        with tf.name_scope("SelectRCNBoxes"):
            def random_select(boxes,boxes_regs,labels,scores,size):
                with tf.name_scope("random_select"):
                    data_nr = tf.shape(labels)[0]
                    indexs = tf.range(data_nr)
                    indexs = wpad(indexs,[0,size-data_nr])
                    indexs = tf.random_shuffle(indexs)
                    indexs = tf.random_crop(indexs,[size])
                    boxes = tf.gather(boxes,indexs)
                    boxes_regs = tf.gather(boxes_regs,indexs)
                    labels = tf.gather(labels,indexs)
                    scores = tf.gather(scores,indexs)
                    return boxes,boxes_regs,labels,scores

            def selectRCNBoxesM0():
                with tf.name_scope("M0"):
                    n_mask = tf.equal(labels,0)
                    p_mask = tf.logical_not(n_mask)
                    n_labels = tf.boolean_mask(labels,n_mask)
                    n_boxes = tf.boolean_mask(boxes,n_mask)
                    n_boxes_regs = tf.boolean_mask(boxes_regs,n_mask)
                    n_scores = tf.boolean_mask(scores,n_mask)
                    p_labels = tf.boolean_mask(labels,p_mask)
                    p_boxes = tf.boolean_mask(boxes,p_mask)
                    p_boxes_regs = tf.boolean_mask(boxes_regs,p_mask)
                    p_scores = tf.boolean_mask(scores,p_mask)
                    p_boxes,p_boxes_regs,p_labels,p_scores = random_select(p_boxes,p_boxes_regs,p_labels,p_scores,pos_nr)
                    n_boxes,n_boxes_regs,n_labels,n_scores = random_select(n_boxes,n_boxes_regs,n_labels,n_scores,neg_nr)
                    
                    return tf.concat([p_boxes,n_boxes],axis=0),tf.concat([p_boxes_regs,n_boxes_regs],axis=0),\
                           tf.concat([p_labels,n_labels],axis=0),tf.concat([p_scores,n_scores],axis=0)

            def selectRCNBoxesM1():
                with tf.name_scope("M1"):
                    return random_select(boxes,boxes_regs, labels, scores,nr)

            def selectRCNBoxesM2():
                with tf.name_scope("M2"):
                    n_mask = tf.equal(labels,0)
                    p_mask = tf.logical_not(n_mask)
                    n_labels = tf.boolean_mask(labels,n_mask)
                    n_boxes = tf.boolean_mask(boxes,n_mask)
                    n_boxes_regs = tf.boolean_mask(boxes_regs,n_mask)
                    n_scores = tf.boolean_mask(scores,n_mask)
                    p_labels = tf.boolean_mask(labels,p_mask)
                    p_boxes = tf.boolean_mask(boxes,p_mask)
                    p_boxes_regs = tf.boolean_mask(boxes_regs,p_mask)
                    p_scores = tf.boolean_mask(scores,p_mask)
                    with tf.name_scope("Select"):
                        n_boxes,n_boxes_regs,n_labels,n_scores =\
                            random_select(n_boxes,n_boxes_regs,n_labels,n_scores,nr-total_pos_nr)
                    return tf.concat([p_boxes,n_boxes],axis=0),tf.concat([p_boxes_regs,n_boxes_regs],axis=0),tf.concat([p_labels,n_labels],axis=0),tf.concat([p_scores,n_scores],axis=0)
            def selectRCNBoxesM3():
                with tf.name_scope("M3"):
                    boxes = tf.constant([[0.0,0.0,0.001,0.001]],dtype=tf.float32)*tf.ones([nr,4],dtype=tf.float32)
                    boxes_regs = tf.zeros_like(boxes,dtype=tf.float32)
                    labels = tf.constant([0])*tf.ones([nr],dtype=tf.int32)
                    scores = tf.ones_like(labels,dtype=tf.float32)
                    return boxes,boxes_regs,labels,scores

            r_boxes,r_boxes_regs,r_labels,r_scores = tf.case({
                tf.logical_and(total_pos_nr>=pos_nr,total_neg_nr>=neg_nr):selectRCNBoxesM0,
                tf.logical_and(tf.logical_and(total_pos_nr<pos_nr,total_neg_nr>=neg_nr),total_pos_nr>0): selectRCNBoxesM2,
                tf.equal(tf.shape(labels)[0],0):selectRCNBoxesM3
            },
                default=selectRCNBoxesM1,
                exclusive=True)
            r_boxes.set_shape([nr,4])
            r_boxes_regs.set_shape([nr,4])
            r_labels.set_shape([nr])
            r_scores.set_shape([nr])
            return r_boxes,r_boxes_regs,r_labels,r_scores

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
    用于计算RPN网络的损失
    '''
    def getRPNLoss(self):
        return losses.od_loss(gregs=self.rpn_gtregs,
                   glabels=self.rpn_gtlabels,
                   classes_logits=self.rpn_logits,
                   bboxes_regs=self.rpn_regs,
                   num_classes=2,
                   reg_loss_weight=self.reg_loss_weight,
                   bboxes_remove_indices=self.anchor_remove_indices,
                   scope="RPNLoss",
                   classes_wise=False)
    '''
    用于计算RCN网络的损失
    '''
    def getRCNLoss(self,labels=None,neg_multiplier=1.0,use_scores=True):
        if labels is None:
            labels = self.rcn_gtlabels
        if use_scores:
            return losses.od_lossv2(gregs=self.rcn_gtregs,
                       glabels=labels,
                       scores=self.rcn_gtscores,
                       classes_logits=self.rcn_logits,
                       bboxes_regs=self.rcn_regs,
                       num_classes=self.num_classes,
                       reg_loss_weight=self.reg_loss_weight,
                       scope="RCNLoss",
                       neg_multiplier=neg_multiplier,
                       classes_wise=True)
        else:
            return losses.od_loss(gregs=self.rcn_gtregs,
                       glabels=labels,
                       classes_logits=self.rcn_logits,
                       bboxes_regs=self.rcn_regs,
                       num_classes=self.num_classes,
                       reg_loss_weight=self.reg_loss_weight,
                       scope="RCNLoss",
                       neg_multiplier=neg_multiplier,
                       classes_wise=True)


    def getProposalBoxes(self,k=1000,threshold=0.5,nms_threshold=0.1):
        with tf.variable_scope("RPNProposalBoxes"):
            self.proposal_boxes,labels,self.proposal_boxes_prob,_ =\
                od.get_prediction(class_prediction=tf.nn.softmax(self.rpn_logits),
                                     bboxes_regs=self.rpn_regs,
                                     proposal_bboxes=self.anchors,
                                     threshold=threshold,
                                     nms_threshold=nms_threshold,
                                     candiate_nr=k,
                                     classes_wise=False)
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
        return self.proposal_boxes,self.proposal_boxes_prob

    '''
    get exactly k proposal boxes, the excess boxes was remove by probability.
    k: output boxes number
    '''
    def getProposalBoxesV3(self,k=1000):
        with tf.variable_scope("RPNProposalBoxes"):
            self.proposal_boxes,labels,self.proposal_boxes_prob =\
                od.get_proposal_boxesv2(class_prediction=tf.nn.softmax(self.rpn_logits),
                                     bboxes_regs=self.rpn_regs,
                                     proposal_bboxes=self.anchors,
                                     candiate_nr=k,
                                     classes_wise=False)
        return self.proposal_boxes,self.proposal_boxes_prob

    '''
    get exactly k proposal boxes, the excess boxes was remove by heristic method.
    k: output boxes number
    nms_threshold: NMS operation threshold.
    '''
    def getProposalBoxesV4(self,k=1000,nms_threshold=0.8):
        with tf.variable_scope("RPNProposalBoxes"):
            self.proposal_boxes,labels,self.proposal_boxes_prob = \
                od.get_proposal_boxesv3(class_prediction=tf.nn.softmax(self.rpn_logits),
                                     bboxes_regs=self.rpn_regs,
                                     proposal_bboxes=self.anchors,
                                     candiate_nr=k,
                                     nms_threshold=nms_threshold,
                                     classes_wise=False)
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
                probs = tf.nn.softmax(self.rcn_logits)
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
        return self.finally_boxes,self.finally_boxes_label,self.finally_boxes_prob

    '''
    process the situation of batch_size greater than one, target boxes number of each imag is very different, so the 
    boxes number of each image is return by lens
    use_soft_nms:是否使用softnms,使用soft nms与不使用soft nms时, nms_threshold的意义有很大的区别， 不使用soft nms时，nms_threshold表示
IOU小于nms_threshold的两个bbox为不同目标，使用soft nms时，nms_threshold表示得分高于nms_threshold的才是真目标
    '''
    def getBoxesV2(self,k=1000,threshold=0.5,nms_threshold=0.1,proposal_boxes=None,limits=None,
                   adjust_probability=None,classes_wise_nms=True,
                   use_soft_nms=False):
        if proposal_boxes is None:
            proposal_boxes = self.proposal_boxes
        with tf.device("/cpu:0"):
            with tf.variable_scope("RCNGetBoxes"):
                probs = tf.nn.softmax(self.rcn_logits)
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
                                         nms_threshold=nms_threshold,
                                         limits=limits,
                                         candiate_nr=k,
                                         classes_wise=True,
                                         classes_wise_nms=classes_wise_nms,
                                         use_soft_nms=use_soft_nms)
        return self.finally_boxes,self.finally_boxes_label,self.finally_boxes_prob,self.rcn_bboxes_lens

    '''
    process the situation of batch_size greater than one, target boxes number of each imag is specified by k
    '''
    def getBoxesV3(self,k=1000,nms_threshold=0.1,proposal_boxes=None,limits=None,
                   adjust_probability=None,classes_wise_nms=True):
        if proposal_boxes is None:
            proposal_boxes = self.proposal_boxes
        with tf.device("/cpu:0"):
            with tf.variable_scope("RCNGetBoxes"):
                probs = tf.nn.softmax(self.rcn_logits)
                if adjust_probability is not None:
                    probs = wnnl.probability_adjust(probs=probs,classes=adjust_probability)
                self.finally_boxes,self.finally_boxes_label,self.finally_boxes_prob =\ 
                    od.get_predictionv3(class_prediction=probs,
                                         bboxes_regs=self.rcn_regs,
                                         proposal_bboxes=proposal_boxes,
                                         nms_threshold=nms_threshold,
                                         limits=limits,
                                         candiate_nr=k,
                                         classes_wise=True,
                                         classes_wise_nms=classes_wise_nms)
        return self.finally_boxes,self.finally_boxes_label,self.finally_boxes_prob

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


