#coding=utf-8
from abc import ABCMeta, abstractmethod
import object_detection.fasterrcnn as fasterrcnn
import tensorflow as tf
import wml_tfutils as wmlt
import object_detection.wlayers as odl

class MaskRCNN(fasterrcnn.FasterRCNN):
    def __init__(self,num_classes,input_shape,batch_size=1):
        super().__init__(num_classes,input_shape,batch_size)
        self.train_mask = False
        #[X,h,w]
        self.mask_logits = None
        #[X,h,w]
        self.finally_mask = None

    '''
    labels:[batch_size*box_nr]
    pmask:[batch_size*box_nr]
    output:[batch_size*pbox_nr,M,N]
    '''
    def buildMaskBranch(self,pmask=None,labels=None,size=(33,33),reuse=False,net=None):
        if net is None:
            net = self.ssbp_net

        if self.train_mask and pmask is None:
            pmask = tf.greater(self.rcn_gtlabels,0)
            pmask = tf.reshape(pmask,[-1])

        if labels is None:
            labels = tf.reshape(self.rcn_gtlabels,[-1])

        if pmask is not None:
            pmask = wmlt.assert_equal(pmask,[tf.shape(net)[:1],tf.shape(pmask)])
            net = tf.boolean_mask(net,pmask)
            labels = tf.boolean_mask(labels,pmask)

        net = tf.image.resize_bilinear(net,size)
        net = self._maskFeatureExtractor(net,reuse=reuse)
        net = tf.transpose(net,perm=(0,3,1,2))
        assert net.get_shape().as_list()[1]==self.num_classes-1,"Error dim size."
        self.mask_logits = wmlt.batch_gather(net,labels-1)
        return self.mask_logits
    '''
    labels:[batch_size*box_nr]
    pmask:[batch_size*box_nr]
    output:[batch_size*pbox_nr,M,N]
    '''
    def buildMaskBranchV2(self,pmask=None,labels=None,bin_size=11,size=(33,33),reuse=False,roipooling=odl.DFROI()):
        base_net = self.base_net
        batch_index, batch_size, box_nr = self.rcn_batch_index_helper(self.proposal_boxes)
        net = roipooling(base_net, self.proposal_boxes, batch_index, bin_size, bin_size)
        net_channel = net.get_shape().as_list()[-1]
        net = wmlt.reshape(net, [batch_size * box_nr, bin_size, bin_size, net_channel])

        if self.train_mask and pmask is None:
            pmask = tf.greater(self.rcn_gtlabels,0)
            pmask = tf.reshape(pmask,[-1])

        if labels is None:
            labels = tf.reshape(self.rcn_gtlabels,[-1])

        if pmask is not None:
            pmask = wmlt.assert_equal(pmask,[tf.shape(net)[:1],tf.shape(pmask)])
            net = tf.boolean_mask(net,pmask)
            labels = tf.boolean_mask(labels,pmask)

        net = tf.image.resize_bilinear(net,size)
        net = self._maskFeatureExtractor(net,reuse=reuse)
        net = tf.transpose(net,perm=(0,3,1,2))
        assert net.get_shape().as_list()[1]==self.num_classes-1,"Error dim size."
        self.mask_logits = wmlt.batch_gather(net,labels-1)
        return self.mask_logits

    '''
    output:[batch_size,X,H,W]
    '''
    def getBoxesAndMask(self,k=1000,mask_threshold=0.5,box_threshold=0.5,proposal_boxes=None,limits=None,
                   adjust_probability=None,nms=None,reuse=False,
                        size=(33,33)
                   ):
        self.getBoxesV2(k=k,
                        threshold=box_threshold,
                        proposal_boxes=proposal_boxes,
                        limits=limits,
                        adjust_probability=adjust_probability,
                        nms=nms)
        max_len = tf.maximum(1,tf.reduce_max(self.rcn_bboxes_lens))
        ssbp_net = wmlt.batch_gather(self.get_5d_ssbp_net(),self.finally_indices[:,:max_len])
        ssbp_net = self.to_4d_ssbp_net(ssbp_net)
        labels = self.finally_boxes_label[:,:max_len]
        labels = tf.reshape(labels,[-1])
        logits = self.buildMaskBranch(labels=labels,size=size,reuse=reuse,net=ssbp_net)
        mask = tf.greater(tf.sigmoid(logits),mask_threshold)
        mask = tf.cast(mask,tf.int32)
        shape = mask.get_shape().as_list()[1:]
        mask = wmlt.reshape(mask,[self.rcn_batch_size,max_len]+shape)
        self.finally_mask = mask
        return self.finally_mask

    def buildFakeMaskBranch(self):
        pmask = tf.ones(tf.shape(self.ssbp_net)[:1],dtype=tf.bool)
        labels = tf.ones(tf.shape(self.ssbp_net)[:1],dtype=tf.int32)
        self.buildMaskBranch(pmask,labels,size=[7,7])

    def buildFakeMaskBranchV2(self):
        pmask = tf.ones(tf.shape(self.ssbp_net)[:1],dtype=tf.bool)
        labels = tf.ones(tf.shape(self.ssbp_net)[:1],dtype=tf.int32)
        self.buildMaskBranchV2(pmask,labels,size=[7,7])


    '''
    net:[batch_size*box_nr,bin_size,bin_size,net_channel]
    output: [batch_size*box_nr,bin_size,bin_size,num_classes]
    '''
    @abstractmethod
    def _maskFeatureExtractor(self,net,reuse=False):
        pass


    '''
    gtmasks:[batch_size,X,H,W]
    gtbboxes:[batch_size,X,4]
    '''
    def getMaskLoss(self,gtbboxes,gtmasks,b_len,gtlabels=None):
        max_boxes_nr = gtbboxes.get_shape().as_list()[1]
        shape = self.mask_logits.get_shape().as_list()
        batch_index, batch_size, box_nr = self.rcn_batch_index_helper(gtbboxes)
        pmask = tf.greater(self.rcn_gtlabels,0)
        #pmask = wmlt.assert_shape_equal(pmask,[pmask,self.rcn_anchor_to_gt_indices])
        gtmasks = tf.expand_dims(gtmasks,axis=-1)
        wmlt.image_summaries(gtmasks[0][:b_len[0],:,:,:],"mask0_1")
        gtmasks = wmlt.tf_crop_and_resize(gtmasks,gtbboxes,shape[1:3])

        wmlt.image_summaries(gtmasks[0][:b_len[0],:,:,:],"mask0")
        gtmasks = tf.squeeze(gtmasks,axis=-1)
        rcn_anchor_to_gt_indices = self.rcn_anchor_to_gt_indices
        rcn_anchor_to_gt_indices = tf.clip_by_value(rcn_anchor_to_gt_indices,0,max_boxes_nr)
        gtmasks = wmlt.batch_gather(gtmasks,rcn_anchor_to_gt_indices)

        gtmasks = tf.reshape(gtmasks,[-1]+gtmasks.get_shape().as_list()[2:])
        pmask = tf.reshape(pmask,[-1])
        if gtlabels is not None:
            gtlabels = wmlt.batch_gather(gtlabels,rcn_anchor_to_gt_indices)
            gtlabels = tf.cast(tf.reshape(gtlabels,[-1]),tf.int32)
            cgtlabels = tf.cast(tf.reshape(self.rcn_gtlabels,[-1]),tf.int32)
            gtmasks = wmlt.assert_equal(gtmasks,[tf.boolean_mask(gtlabels,pmask),tf.boolean_mask(cgtlabels,pmask)],"ASSERT_GTLABELS_EQUAL")
        gtmasks = tf.boolean_mask(gtmasks,pmask)
        log_mask  = tf.expand_dims(gtmasks,axis=-1)
        wmlt.image_summaries(log_mask,"mask")
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gtmasks,logits=self.mask_logits)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar("mask_loss",loss)
        tf.losses.add_loss(loss)
        return loss
