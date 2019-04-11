#coding=utf-8
from abc import ABCMeta, abstractmethod
import object_detection.fasterrcnn as fasterrcnn
import tensorflow as tf
import wml_tfutils as wmlt

class MaskRCNN(fasterrcnn.FasterRCNN):
    def __init__(self,num_classes,input_shape,batch_size=1):
        super().__init__(num_classes,input_shape,batch_size)
        self.train_mask = False
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
    '''
    def getMaskLoss(self,gtbboxes,gtmasks):
        shape = self.mask_logits.get_shape().as_list()
        #gtmasks = tf.transpose(gtmasks,perm=(0,2,3,1))
        rcn_anchor_to_gt_indices = self.rcn_anchor_to_gt_indices
        gtmasks = wmlt.batch_gather(gtmasks,rcn_anchor_to_gt_indices)
        bboxes = wmlt.batch_gather(gtbboxes,rcn_anchor_to_gt_indices)
        batch_index, batch_size, box_nr = self.rcn_batch_index_helper(bboxes)
        bboxes = tf.reshape(bboxes,[-1,4])
        batch_index = tf.reshape(batch_index,[-1])
        gtmasks = tf.image.crop_and_resize(image=gtmasks, boxes=bboxes, box_ind=batch_index, crop_size=shape[1:3])
        #gtmasks = tf.concat([tf.zeros_like(gtmasks),gtmasks],axis=3)

        pmask = tf.greater(self.rcn_gtlabels,0)
        pmask = wmlt.assert_shape_equal(pmask,[pmask,self.rcn_anchor_to_gt_indices])
        rcn_anchor_to_gt_indices = tf.boolean_mask(self.rcn_anchor_to_gt_indices,pmask)
        gtmasks = tf.cast(gtmasks,tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gtmasks,logits=self.mask_logits)
        loss = tf.reduce_mean(loss)
        tf.losses.add_loss(loss)
        return loss
