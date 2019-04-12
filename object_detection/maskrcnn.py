#coding=utf-8
from abc import ABCMeta, abstractmethod
import object_detection.fasterrcnn as fasterrcnn
import tensorflow as tf
import wml_tfutils as wmlt

class MaskRCNN(fasterrcnn.FasterRCNN):
    def __init__(self,num_classes,input_shape,batch_size=1):
        super().__init__(num_classes,input_shape,batch_size)
        self.train_mask = False
        #[X,h,w]
        self.mask_logits = None

    '''
    labels:[batch_size*box_nr]
    pmask:[batch_size*box_nr]
    output:[batch_size*pbox_nr,M,N]
    '''
    def buildMaskBranch(self,pmask=None,labels=None,size=(33,33)):
        if self.train_mask and pmask is None:
            pmask = tf.greater(self.rcn_gtlabels,0)
            pmask = tf.reshape(pmask,[-1])
        pmask = wmlt.assert_equal(pmask,[tf.shape(self.ssbp_net)[:1],tf.shape(pmask)])
        net = tf.boolean_mask(self.ssbp_net,pmask)
        net = tf.image.resize_bilinear(net,size)
        net = self._maskFeatureExtractor(net)
        if labels is None:
            labels = tf.reshape(self.rcn_gtlabels,[-1])
        labels = tf.boolean_mask(labels,pmask)
        net = tf.transpose(net,perm=(0,3,1,2))
        self.mask_logits = wmlt.batch_gather(net,labels-1)
        return self.mask_logits

    '''
    net:[batch_size*box_nr,bin_size,bin_size,net_channel]
    output: [batch_size*box_nr,bin_size,bin_size,num_classes]
    '''
    @abstractmethod
    def _maskFeatureExtractor(self,net):
        pass


    '''
    gtmasks:[batch_size,X,H,W]
    gtbboxes:[batch_size,X,4]
    '''
    def getMaskLoss(self,gtbboxes,gtmasks,b_len):
        max_boxes_nr = gtbboxes.get_shape().as_list()[1]
        shape = self.mask_logits.get_shape().as_list()
        batch_index, batch_size, box_nr = self.rcn_batch_index_helper(gtbboxes)
        pmask = tf.greater(self.rcn_gtlabels,0)
        pmask = wmlt.assert_shape_equal(pmask,[pmask,self.rcn_anchor_to_gt_indices])
        pmask = tf.Print(pmask,[pmask,tf.reduce_max(self.rcn_anchor_to_gt_indices),tf.reduce_min(self.rcn_anchor_to_gt_indices)],"pmask",first_n=64)
        gtmasks = tf.expand_dims(gtmasks,axis=-1)
        gtmasks = wmlt.tf_crop_and_resize(gtmasks,gtbboxes,shape[1:3])
        for i in range(batch_size):
            gtmasks = tf.Print(gtmasks,[b_len[i],tf.reduce_max(self.rcn_anchor_to_gt_indices[i])],"test0")

        wmlt.image_summaries(gtmasks[:b_len[0],:,:,:],"mask0")
        gtmasks = tf.squeeze(gtmasks,axis=-1)
        rcn_anchor_to_gt_indices = self.rcn_anchor_to_gt_indices
        rcn_anchor_to_gt_indices = tf.clip_by_value(rcn_anchor_to_gt_indices,0,max_boxes_nr)
        pmask = tf.Print(pmask,[pmask,tf.reduce_max(rcn_anchor_to_gt_indices),tf.reduce_min(rcn_anchor_to_gt_indices)],"pmask2",first_n=64)
        gtmasks = wmlt.reshape(gtmasks,[batch_size,box_nr]+gtmasks.get_shape().as_list()[1:])
        gtmasks = wmlt.batch_gather(gtmasks,rcn_anchor_to_gt_indices)

        gtmasks = tf.reshape(gtmasks,[-1]+gtmasks.get_shape().as_list()[2:])
        pmask = tf.reshape(pmask,[-1])
        gtmasks = tf.boolean_mask(gtmasks,pmask)
        log_mask  = tf.expand_dims(gtmasks,axis=-1)
        wmlt.image_summaries(log_mask,"mask")
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gtmasks,logits=self.mask_logits)
        loss = tf.reduce_mean(loss)
        tf.losses.add_loss(loss)
        return loss
