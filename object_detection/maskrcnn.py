#coding=utf-8
from abc import ABCMeta, abstractmethod
import object_detection.fasterrcnn as fasterrcnn
import tensorflow as tf
import wml_tfutils as wmlt
import object_detection.wlayers as odl
import object_detection.od_toolkit as od
import object_detection.utils as odu
from wtfop.wtfop_ops import wpad
import img_utils as wmli

class MaskRCNN(fasterrcnn.FasterRCNN):
    def __init__(self,num_classes,input_shape,batch_size=1,loss_scale=10.0):
        super().__init__(num_classes,input_shape,batch_size)
        self.train_mask = False
        #[X,h,w]
        self.mask_logits = None
        #[X,h,w]
        self.finally_mask = None
        self.loss_scale = loss_scale

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
    labels:[batch_size*box_nr]
    pmask:[batch_size*box_nr]
    output:[batch_size*pbox_nr,M,N]
    '''
    def buildMaskBranchV3(self,gtbboxes,gtlabels,gtlens,bboxes_nr,net=None,net_process=None,bin_size=11,size=(33,33),reuse=False,roipooling=odl.DFROI()):
        def random_select(labels,data_nr):
            with tf.name_scope("random_select"):
                if labels.dtype != tf.int32:
                    labels = tf.cast(labels,tf.int32)
                size = tf.shape(labels)[0]
                indexs = tf.range(data_nr)
                indexs = wpad(indexs, [0, size - data_nr])
                indexs = tf.random_shuffle(indexs)
                indexs = tf.random_crop(indexs, [bboxes_nr])
                labels = tf.gather(labels, indexs)
                return labels,indexs
        def batch_random_select(labels,data_nr):
            return tf.map_fn(lambda x:random_select(x[0],x[1]),elems=(labels,data_nr),dtype=(tf.int32,tf.int32),back_prop=False)
        gtlabels,indices = batch_random_select(gtlabels,gtlens)
        gtbboxes = wmlt.batch_gather(gtbboxes,indices)
        if net is None:
            net = self.base_net
        batch_index, batch_size, box_nr = self.rcn_batch_index_helper(gtbboxes)
        net = roipooling(net, gtbboxes, batch_index, bin_size, bin_size)
        net_channel = net.get_shape().as_list()[-1]
        net = wmlt.reshape(net, [batch_size * box_nr, bin_size, bin_size, net_channel])
        if net_process is not None:
            net = net_process(net)
        net = tf.image.resize_bilinear(net,size)
        net = self._maskFeatureExtractor(net,reuse=reuse)
        net = tf.transpose(net,perm=(0,3,1,2))
        assert net.get_shape().as_list()[1]==self.num_classes-1,"Error dim size."
        gtlabels = tf.reshape(gtlabels,[-1])
        self.mask_logits = wmlt.batch_gather(net,gtlabels-1)
        return self.mask_logits,gtlabels,gtbboxes,indices

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

    def buildFakeMaskBranch(self,net=None):
        pmask = tf.ones(tf.shape(self.ssbp_net)[:1],dtype=tf.bool)
        labels = tf.ones(tf.shape(self.ssbp_net)[:1],dtype=tf.int32)
        self.buildMaskBranch(pmask,labels,size=[7,7],net=net)

    def buildFakeMaskBranchV2(self,net=None):
        pmask = tf.ones(tf.shape(self.ssbp_net)[:1],dtype=tf.bool)
        labels = tf.ones(tf.shape(self.ssbp_net)[:1],dtype=tf.int32)
        self.buildMaskBranchV2(pmask,labels,size=[7,7],net=net)


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
    def getMaskLoss(self,gtbboxes,gtmasks,gtlens,gtlabels=None):
        max_boxes_nr = gtbboxes.get_shape().as_list()[1]
        shape = self.mask_logits.get_shape().as_list()
        pmask = tf.greater(self.rcn_gtlabels,0)
        #pmask = wmlt.assert_shape_equal(pmask,[pmask,self.rcn_anchor_to_gt_indices])
        gtmasks = tf.expand_dims(gtmasks,axis=-1)
        wmlt.image_summaries(gtmasks[0][:gtlens[0],:,:,:],"mask0_1")
        gtmasks = wmlt.tf_crop_and_resize(gtmasks,gtbboxes,shape[1:3])

        wmlt.image_summaries(gtmasks[0][:gtlens[0],:,:,:],"mask0")
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
        tf.losses.add_loss(loss*self.loss_scale)
        return loss

    def getMaskLossV1(self,gtmasks,gtlabels=None):
        shape = self.mask_logits.get_shape().as_list()
        pmask = tf.greater(self.rcn_gtlabels,0)

        rcn_anchor_to_gt_indices = self.rcn_anchor_to_gt_indices
        rcn_anchor_to_gt_indices = tf.maximum(rcn_anchor_to_gt_indices,0)
        gtmasks = wmlt.batch_gather(gtmasks,rcn_anchor_to_gt_indices)
        bboxes = self.getRCNBoxes()
        gtmasks = tf.expand_dims(gtmasks,axis=-1)
        gtmasks = wmlt.tf_crop_and_resize(gtmasks,bboxes,shape[1:3])
        gtmasks = tf.squeeze(gtmasks,axis=-1)

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
        tf.losses.add_loss(loss*self.loss_scale)
        return loss

    def getMaskLossV2(self,gtbboxes,gtmasks,indices):
        shape = self.mask_logits.get_shape().as_list()

        gtmasks = wmlt.batch_gather(gtmasks,indices)
        gtmasks = tf.expand_dims(gtmasks,axis=-1)
        org_mask = tf.identity(gtmasks)
        org_mask = tf.reshape(org_mask,[-1]+org_mask.get_shape().as_list()[2:])
        gtmasks = wmlt.tf_crop_and_resize(gtmasks,gtbboxes,shape[1:3])
        gtmasks = tf.squeeze(gtmasks,axis=-1)

        gtmasks = tf.reshape(gtmasks,[-1]+gtmasks.get_shape().as_list()[2:])
        log_mask  = tf.expand_dims(gtmasks,axis=-1)
        log_boxes = tf.expand_dims(tf.reshape(gtbboxes.[-1,4]),axis=1)
        log_mask1 = odu.tf_draw_image_with_box(org_mask,log_boxes,scale=False)
        log_mask = wmli.concat_images([log_mask1,log_mask])
        wmlt.image_summaries(log_mask,"mask",max_outputs=40)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gtmasks,logits=self.mask_logits)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar("mask_loss",loss)
        tf.losses.add_loss(loss*self.loss_scale)
        return loss

    def getRCNBoxes(self):
        probs = tf.nn.softmax(self.rcn_logits)
        boxes,_,_  = od.get_predictionv4(class_prediction=probs,bboxes_regs=self.rcn_regs,
                                        proposal_bboxes=self.proposal_boxes,classes_wise=self.pred_bboxes_classwise)
        return tf.stop_gradient(boxes)

    def getLoss(self,gtbboxes,gtmasks,gtlens,gtlabels=None,use_scores=False):
        pc_loss, pr_loss, nc_loss, psize_div_all = self.getRCNLoss(use_scores=use_scores)
        mask_loss = self.getMaskLoss(gtbboxes=gtbboxes, gtmasks=gtmasks,gtlens=gtlens,gtlabels=gtlabels)
        return mask_loss,pc_loss, pr_loss, nc_loss, psize_div_all

    def getLossV1(self,gtmasks,gtlabels=None,use_scores=False):
        pc_loss, pr_loss, nc_loss, psize_div_all = self.getRCNLoss(use_scores=use_scores)
        mask_loss = self.getMaskLossV1(gtmasks=gtmasks,gtlabels=gtlabels)
        return mask_loss,pc_loss, pr_loss, nc_loss, psize_div_all

    '''
    use buildMaskBranchV3/buildFakeMaskBranch for mask branch
    in this 
    '''
    def getLossV2(self,gtbboxes,gtmasks,indices,use_scores=False):
        pc_loss, pr_loss, nc_loss, psize_div_all = self.getRCNLoss(use_scores=use_scores)
        mask_loss = self.getMaskLossV2(gtbboxes=gtbboxes,gtmasks=gtmasks,indices=indices)
        return mask_loss,pc_loss, pr_loss, nc_loss, psize_div_all
