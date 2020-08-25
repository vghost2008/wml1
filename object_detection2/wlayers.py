#coding=utf-8
import tensorflow as tf
from wtfop.wtfop_ops import roi_pooling,boxes_nms,decode_boxes1
from wtfop.wtfop_ops import boxes_relative_to_absolute
import object_detection2.bboxes as odb
import wml_tfutils as wmlt
import wtfop.wtfop_ops as wop
from basic_tftools import channel
import object_detection2.od_toolkit as odt
import wnnlayer as wnnl

slim=tf.contrib.slim

'''
bboxes:[batch_size,box_nr,box_dim]
output:[batch_size,box_nr],batch index for each box
'''
def _make_batch_index_for_pooler(bboxes):
    with tf.name_scope("make_batch_index_for_pooler"):
        batch_size,box_nr,box_dim = wmlt.combined_static_and_dynamic_shape(bboxes)
        batch_index = tf.expand_dims(tf.range(batch_size), axis=1) * tf.ones([1, box_nr], dtype=tf.int32)
        return batch_index

class WROIPool(object):
    def __init__(self,bin_size=[2,2],output_size=[7,7]):
        self.output_size = output_size
        self.bin_size = list(bin_size)

    def __call__(self,net,bboxes):
        '''
        bboxes [batch_size,X,4] (ymin,xmin,ymax,xmax)相对坐标
        net:[batch_size,X,4]
        输出：[Y,pool_height,pool_width,num_channels] //num_channels为fmap的通道数, Y=batch_size*X
        '''
        with tf.name_scope("WROIPooling"):
            batch_size,H,W,C = wmlt.combined_static_and_dynamic_shape(net)
            _,box_nr,box_dim = wmlt.combined_static_and_dynamic_shape(bboxes)
            batch_index = _make_batch_index_for_pooler(bboxes)
            batch_index = tf.reshape(batch_index,[-1,1])
            batch_index = tf.cast(batch_index,tf.float32)
            bboxes = tf.reshape(bboxes,[-1,4])
            absolute_bboxes = boxes_relative_to_absolute(bboxes, width=W, height=H)
            absolute_bboxes = tf.transpose(absolute_bboxes)
            absolute_bboxes = tf.gather(absolute_bboxes, [1, 0, 3, 2])
            absolute_bboxes = tf.transpose(absolute_bboxes)
            bboxes_with_batch_index = tf.concat(values=[batch_index, absolute_bboxes], axis=1)
            pool_height,pool_width = self.output_size
            width = pool_width*self.bin_size[1]
            height = pool_height*self.bin_size[0]
            net = roi_pooling(net, bboxes_with_batch_index, pool_height=height, pool_width=width)
            net = tf.reshape(net,[batch_size*box_nr,height,width,C])
            if self.bin_size[0]>1 and self.bin_size[1]>1:
                net = tf.nn.max_pool(net, ksize=[1] + self.bin_size + [1], strides=[1] + self.bin_size + [1],
                                     padding="SAME")
            return net
'''
实现ROI Align in mask-RCNN
'''
class WROIAlign:
    '''
    bin_size:(h,w)每一个格子的大小

    '''
    def __init__(self,bin_size=[2,2],output_size=[7,7]):
        assert(len(bin_size)>=2)
        self.bin_size = list(bin_size)
        self.output_size = output_size
    '''
    bboxes:[batch_size,X,4]
    net:[batch_size,H,W,C]
    输出：[Y,pool_height,pool_width,num_channels] //num_channels为fmap的通道数, Y=batch_size*X
    '''
    def __call__(self, net,bboxes):
        with tf.variable_scope("WROIAlign"):
            batch_index = _make_batch_index_for_pooler(bboxes)
            pool_height,pool_width = self.output_size
            if not isinstance(bboxes,tf.Tensor):
                bboxes = tf.convert_to_tensor(bboxes,dtype=tf.float32)
            bboxes = tf.reshape(bboxes,[-1,4])
            width = pool_width*self.bin_size[1]
            height = pool_height*self.bin_size[0]
            batch_index = tf.stop_gradient(tf.reshape(batch_index, [-1]))
            bboxes = tf.stop_gradient(tf.reshape(bboxes, [-1, 4]))
            net = tf.image.crop_and_resize(image=net, boxes=bboxes, box_ind=batch_index, crop_size=[height, width])
            if self.bin_size[0]>1 and self.bin_size[1]>1:
                net = tf.nn.max_pool(net, ksize=[1] + self.bin_size + [1], strides=[1] + self.bin_size + [1],
                                 padding="SAME")
            return net

class WROIAlignRotated:
    '''
    bin_size:(h,w)每一个格子的大小

    '''
    def __init__(self,bin_size=[2,2],output_size=[7,7]):
        assert(len(bin_size)>=2)
        self.bin_size = list(bin_size)
        self.output_size = output_size
    def make_outbox_and_inbox(self,bboxes):
        raise NotImplementedError("Not implemented.")
        return bboxes,bboxes
    '''
    bboxes:[batch_size,X,5], (ymin,xmin,ymax,xmax,angle(radian))
    net:[batch_size,H,W,C]
    输出：[Y,pool_height,pool_width,num_channels] //num_channels为fmap的通道数, Y=batch_size*X
    '''
    def __call__(self, net,bboxes):
        with tf.variable_scope("WROIAlignRotated"):
            pool_height,pool_width = self.output_size
            if not isinstance(bboxes,tf.Tensor):
                bboxes = tf.convert_to_tensor(bboxes,dtype=tf.float32)
            bboxes = tf.reshape(bboxes,[-1,4])
            width = pool_width*self.bin_size[1]
            height = pool_height*self.bin_size[0]
            batch_index = _make_batch_index_for_pooler(bboxes)
            batch_index = tf.stop_gradient(tf.reshape(batch_index, [-1]))
            bboxes = tf.stop_gradient(tf.reshape(bboxes, [-1, 4]))
            angles = bboxes[...,4]
            angles = tf.reshape(angles,[-1])
            outboxes,inboxes = self.make_outbox_and_inbox(bboxes,)
            net = tf.image.crop_and_resize(image=net, boxes=outboxes, box_ind=batch_index, crop_size=[height*4, width*4])
            net = tf.contrib.image.rotate(net,angles)
            net = tf.image.crop_and_resize(image=net, boxes=inboxes, box_ind=batch_index, crop_size=[height, width])
            if self.bin_size[0]>1 and self.bin_size[1]>1:
                net = tf.nn.max_pool(net, ksize=[1] + self.bin_size + [1], strides=[1] + self.bin_size + [1],
                                     padding="SAME")
            return net

class WROIMultiScale:
    '''
    bin_size:(h,w)每一个格子的大小

    '''
    def __init__(self,bin_size=[2,2],output_size=[7,7]):
        assert(len(bin_size)>=2)
        self.bin_size = list(bin_size)
        self.output_size = output_size
        self.scale = 2.0
        self.pool0 = WROIAlign(bin_size=bin_size,output_size=output_size)
        output_size1 = [x+2*(x//2) for x in output_size]
        self.pool1 = WROIAlign(bin_size=[1,1], output_size=output_size1)

    '''
    bboxes:[batch_size,X,4]
    net:[batch_size,H,W,C]
    输出：[Y,pool_height,pool_width,num_channels] //num_channels为fmap的通道数, Y=batch_size*X
    '''
    def __call__(self, net,bboxes):
        with tf.variable_scope("roi_multi_scale",reuse=tf.AUTO_REUSE):
            conv_kernel = self.output_size[0]
            net0 = self.pool0(net,bboxes)
            net1 = self.pool1(net,odb.scale_bboxes(bboxes,scale=[self.scale,self.scale]))
            normalizer_fn,normalizer_params = odt.get_norm("evo_norm_s0",is_training=True)
            ch = channel(net)
            net1 = slim.separable_conv2d(net1, ch, kernel_size=[conv_kernel, conv_kernel], padding="VALID",
                                            depth_multiplier=1,
                                            normalizer_fn=normalizer_fn,
                                            normalizer_params=normalizer_params,
                                            scope=f"sep_conv")
            net1 = wnnl.non_local_blockv1(net1,
                                          normalizer_fn=normalizer_fn,
                                          normalizer_params=normalizer_params,
                                          activation_fn=None)
            return net0,net1


class WROIKeepRatio:
    '''
    bin_size:(h,w)每一个格子的大小

    '''
    def __init__(self,bin_size=[2,2],output_size=[7,7]):
        assert(len(bin_size)>=2)
        self.bin_size = list(bin_size)
        self.output_size = output_size
    '''
    bboxes:[batch_size,X,4]
    net:[batch_size,H,W,C]
    输出：[Y,pool_height,pool_width,num_channels] //num_channels为fmap的通道数, Y=batch_size*X
    '''
    def __call__(self, net,bboxes):
        with tf.variable_scope("WROIKeepRatio"):
            batch_index = _make_batch_index_for_pooler(bboxes)
            pool_height,pool_width = self.output_size
            if not isinstance(bboxes,tf.Tensor):
                bboxes = tf.convert_to_tensor(bboxes,dtype=tf.float32)
            bboxes = tf.reshape(bboxes,[-1,4])
            width = pool_width*self.bin_size[1]
            height = pool_height*self.bin_size[0]
            batch_index = tf.stop_gradient(tf.reshape(batch_index, [-1]))
            bboxes = tf.stop_gradient(tf.reshape(bboxes, [-1, 4]))
            bboxes,mask = self.get_bboxes_and_mask(bboxes,width,height)
            mask = tf.expand_dims(mask,axis=-1)
            net = tf.image.crop_and_resize(image=net, boxes=bboxes, box_ind=batch_index, crop_size=[height, width])
            net = net*mask
            if self.bin_size[0]>1 and self.bin_size[1]>1:
                net = tf.nn.max_pool(net, ksize=[1] + self.bin_size + [1], strides=[1] + self.bin_size + [1],
                                     padding="SAME")
            return net

    @staticmethod
    @wmlt.add_name_scope
    def get_bboxes_and_mask(bboxes,crop_width,crop_height):
        '''

        :param bboxes: [B,4]
        :param crop_width: ()
        :param crop_height: ()
        :return:
        '''
        with tf.device(":/cpu:0"):
            ymin,xmin,ymax,xmax = bboxes[:,0],bboxes[:,1],bboxes[:,2],bboxes[:,3]
            width = xmax-xmin
            height = ymax-ymin
            size = tf.where(tf.greater(width,height),width,height)
            nxmin = tf.where(tf.less(xmin+size,0.5),xmin,tf.maximum(0.,xmax-size))
            nymin = tf.where(tf.less(ymin+size,0.5),ymin,tf.maximum(0.,ymax-size))
            nxmax = nxmin+size
            nymax = nymin+size
            bboxes = tf.stack([nymin,nxmin,nymax,nxmax],axis=-1)
            xmin_pad = (xmin-nxmin)*(crop_width-1)/size
            xmax_pad = (xmax-nxmin)*(crop_width-1)/size+1.1
            B,_ = wmlt.combined_static_and_dynamic_shape(bboxes)

            def draw_boxes(img,box):
                return wop.fill_bboxes(img,tf.expand_dims(box,axis=0),v=0.0,include_last=False)
            mask = tf.ones([B,crop_height,crop_width])

            tmp_bboxes = tf.stack([tf.zeros([B]),tf.zeros([B]),tf.ones([B])*crop_height,xmin_pad],axis=-1)
            mask = tf.map_fn(lambda x:draw_boxes(x[0],x[1]),elems=(mask,tmp_bboxes),dtype=tf.float32,
                             back_prop=False,parallel_iterations=16)
            tmp_bboxes = tf.stack([tf.zeros([B]),xmax_pad,tf.ones([B])*crop_height,tf.ones([B])*crop_width],axis=-1)
            mask = tf.map_fn(lambda x:draw_boxes(x[0],x[1]),elems=(mask,tmp_bboxes),dtype=tf.float32,
                             back_prop=False,parallel_iterations=16)

            ymin_pad = (ymin-nymin)*(crop_height-1)/size
            ymax_pad = (ymax-nymin)*(crop_height-1)/size+1.1

            tmp_bboxes = tf.stack([tf.zeros([B]),tf.zeros([B]),ymin_pad,tf.ones([B])*crop_width],axis=-1)
            mask = tf.map_fn(lambda x:draw_boxes(x[0],x[1]),elems=(mask,tmp_bboxes),dtype=tf.float32,
                             back_prop=False,parallel_iterations=16)
            tmp_bboxes = tf.stack([ymax_pad,tf.zeros([B]),tf.ones([B])*crop_height,tf.ones([B])*crop_width],axis=-1)
            mask = tf.map_fn(lambda x:draw_boxes(x[0],x[1]),elems=(mask,tmp_bboxes),dtype=tf.float32,
                             back_prop=False,parallel_iterations=16)

            return tf.stop_gradient(bboxes),tf.stop_gradient(mask)

class MixPool:
    def __init__(self,bin_size=[2,2],output_size=[7,7]):
        #self.roi_align_pool = WROIAlign(bin_size=bin_size,output_size=output_size)
        #self.roi_keep_ratio_pool = WROIKeepRatio(bin_size=bin_size,output_size=output_size)
        self.roi_align_pool = WROIAlign(bin_size=[1,1],output_size=[14,14])
        self.bin_size = [2,2]

    def __call__(self, net,bboxes):
        #net0 = self.roi_keep_ratio_pool(net,bboxes)
        #net1 = self.roi_align_pool(net,bboxes)
        #return net0,net1
        net = self.roi_align_pool(net,bboxes)
        net0 = tf.nn.max_pool(net, ksize=[1] + self.bin_size + [1], strides=[1] + self.bin_size + [1],
                             padding="SAME")
        net1 = tf.nn.avg_pool(net, ksize=[1] + self.bin_size + [1], strides=[1] + self.bin_size + [1],
                              padding="SAME")
        return net0,net1


'''
bboxes:[X,4]
labels:[X]
probs:[X]
'''
def boxes_nms(bboxes,labels,probs,threshold=0.5,max_output_size=None,classes_wise=True):
    if max_output_size is None:
        max_output_size = tf.shape(labels)[0]
    indices = tf.image.non_max_suppression(boxes=bboxes,scores=probs,iou_threshold=threshold,max_output_size=max_output_size)
    bboxes = tf.gather(bboxes,indices)
    labels = tf.gather(labels,indices)
    probs = tf.gather(probs,indices)
    return bboxes,labels,probs
'''
get exactly k boxes
policy: first use nms to remove same target, if after nms have less than k targets, put the top k-targets_size(by probability
boxes back to results. 
bboxes:[X,4]
labels:[X]
probs:[X]
'''
def boxes_nms_nr(bboxes,labels,probs,threshold=0.5,k=1000,classes_wise=True):

    data_nr = tf.shape(labels)[0]
    #bboxes = tf.Print(bboxes,[tf.shape(bboxes),tf.shape(probs)])
    #bboxes = tf.Print(bboxes,[bboxes,probs,threshold,k])
    indices = tf.image.non_max_suppression(boxes=bboxes,scores=probs,iou_threshold=threshold,max_output_size=k)
    indices,_ = tf.nn.top_k(-indices,k=tf.shape(indices)[0])
    indices = -indices
    #indices = tf.Print(indices,[tf.shape(indices),indices])

    lmask = tf.sparse_to_dense(sparse_indices=indices,output_shape=[data_nr],sparse_values=1,default_value=0)
    a_op = tf.Assert(k<=data_nr,[k,data_nr],name="boxes_nms_nr_assert")
    ones = tf.ones_like(lmask)

    def less_fn():
        i = tf.constant(0)

        def cond(i,mask):
            return tf.reduce_sum(mask)<k

        def body(i,mask):
            m = tf.sparse_to_dense(sparse_indices=[i],output_shape=tf.shape(mask),sparse_values=True,default_value=False)
            mask = tf.where(m,ones,mask)
            i += 1
            return i,mask

        i, rmask = tf.while_loop(cond=cond,body=body,loop_vars=[i,lmask],back_prop=False)

        return rmask

    with tf.control_dependencies([a_op]):
        mask = tf.cond(tf.reduce_sum(lmask)<k,less_fn,lambda:lmask,"boxes_nms_nr_cond")

    mask = tf.cast(mask,tf.bool)
    bboxes = tf.boolean_mask(bboxes,mask)
    labels = tf.boolean_mask(labels,mask)
    probs = tf.boolean_mask(probs,mask)
    bboxes = tf.reshape(bboxes,[k,4])
    labels = tf.reshape(labels,[k])
    probs = tf.reshape(probs,[k])

    return bboxes,labels,probs

def batch_nms_wrapper(bboxes,classes,lens,confidence=None,nms=None,k=200,sort=False):
    with tf.name_scope("batch_nms_wrapper"):
        if confidence is None:
            confidence = tf.ones_like(classes,dtype=tf.float32)
        def do_nms(sboxes,sclasses,sconfidence,lens):
            if sort:
                r_index = tf.range(lens)
                sconfidence,t_index = tf.nn.top_k(sconfidence[:lens],k=lens)
                sboxes = tf.gather(sboxes,t_index)
                sclasses = tf.gather(sclasses,t_index)
                r_index = tf.gather(r_index,t_index)
            boxes,labels,nms_indexs = nms(bboxes=sboxes[:lens],classes=sclasses[:lens],confidence=sconfidence[:lens])
            r_len = tf.shape(nms_indexs)[0]
            boxes = tf.pad(boxes,[[0,k-r_len],[0,0]])
            labels = tf.pad(labels,[[0,k-r_len]])
            if sort:
                nms_indexs = tf.gather(r_index,nms_indexs)
            nms_indexs = tf.pad(nms_indexs,[[0,k-r_len]])
            return [boxes,labels,nms_indexs,r_len]
        boxes,labels,nms_indexs,lens = wmlt.static_or_dynamic_map_fn(lambda x:do_nms(x[0],x[1],x[2],x[3]),elems=[bboxes,classes,confidence,lens],
                                                                     dtype=[tf.float32,tf.int32,tf.int32,tf.int32])
        return boxes,labels,nms_indexs,lens

'''
paper: Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression
GIoU = IoU - (C - (A U B)) / C
GIoU_Loss = 1 - GIoU
bboxes0: [...,4] (ymin,xmin,ymax,xmax) or (xmin,ymin,xmax,ymax)
bboxes1: [...,4] (ymin,xmin,ymax,xmax) or (xmin,ymin,xmax,ymax)
返回:
[-1,1]
'''
def giou(bboxes0, bboxes1,name=None):
    with tf.name_scope(name,default_name="giou"):
        # 1. calulate intersection over union
        area_1 = (bboxes0[..., 2] - bboxes0[..., 0]) * (bboxes0[..., 3] - bboxes0[..., 1])
        area_2 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])

        intersection_wh = tf.minimum(bboxes0[..., 2:], bboxes1[..., 2:]) - tf.maximum(bboxes0[..., :2], bboxes1[..., :2])
        intersection_wh = tf.maximum(intersection_wh, 0)

        intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
        union = (area_1 + area_2) - intersection

        ious = intersection / tf.maximum(union, 1e-10)

        # 2. (C - (A U B))/C
        C_wh = tf.maximum(bboxes0[..., 2:], bboxes1[..., 2:]) - tf.minimum(bboxes0[..., :2], bboxes1[..., :2])
        C_wh = tf.maximum(C_wh, 1e-10)
        C = C_wh[..., 0] * C_wh[..., 1]

        giou = ious - (C - union) /C
        return giou

def giou_loss(bboxes0, bboxes1,name=None):
    with tf.name_scope(name,default_name="giou_loss"):
        return 1-giou(bboxes0,bboxes1)
