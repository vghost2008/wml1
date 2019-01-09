#coding=utf-8
import tensorflow as tf
from wtfop.wtfop_ops import roi_pooling,boxes_nms,decode_boxes1
from wtfop.wtfop_ops import boxes_relative_to_absolute
import bboxes as odb

slim=tf.contrib.slim

'''
bboxes [X,4] (ymin,xmin,ymax,xmax)相对坐标
输出：[X,pool_height,pool_width,num_channels] //num_channels为fmap的通道数
'''
def wroi_pooling(fmap,bboxes,batch_index,pool_height,pool_width):
    with tf.name_scope("WROIPooling"):
        net_shape = fmap.get_shape().as_list()
        batch_index = tf.reshape(batch_index,[-1,1])
        batch_index = tf.cast(batch_index,tf.float32)
        boxes_shape = bboxes.get_shape().as_list()
        bboxes = tf.reshape(bboxes,[-1,4])
        absolute_bboxes = boxes_relative_to_absolute(bboxes, width=net_shape[2], height=net_shape[1])
        absolute_bboxes = tf.transpose(absolute_bboxes)
        absolute_bboxes = tf.gather(absolute_bboxes, [1, 0, 3, 2])
        absolute_bboxes = tf.transpose(absolute_bboxes)
        bboxes_with_batch_index = tf.concat(values=[batch_index, absolute_bboxes], axis=1)
        y = roi_pooling(fmap, bboxes_with_batch_index, pool_height=pool_height, pool_width=pool_width)
        y = tf.reshape(y,boxes_shape[:2]+[pool_height,pool_width,net_shape[-1]])
        return y
'''
实现ROI Align in mask-RCNN
'''
class WROIAlign:
    '''
    bin_size:(h,w)每一个格子的大小

    '''
    def __init__(self,bin_size=[2,2],multiplier=None):
        assert(len(bin_size)>=2)
        self.bin_size = bin_size
        self.multiplier = multiplier
    '''
    bboxes:[batch_size,nr,4]
    '''
    def __call__(self, fmap,bboxes,batch_index,pool_height,pool_width):
        with tf.variable_scope("WROIAlign"):
            if batch_index.dtype is not tf.int32:
                batch_index = tf.cast(batch_index,tf.int32)
            width = pool_width*self.bin_size[1]
            height = pool_height*self.bin_size[0]
            boxes_shape = bboxes.get_shape().as_list()

            '''if len(boxes_shape)==3 and (boxes_shape[0] > 2 or boxes_shape[0] is None):
                def fun(box,index):
                    if self.multiplier is not None:
                        box = odb.scale_bboxes(box, scale=self.multiplier)
                    net = tf.image.crop_and_resize(image=fmap, boxes=box, box_ind=index, crop_size=[height, width])
                    net = tf.nn.max_pool(net, ksize=[1] + self.bin_size + [1], strides=[1] + self.bin_size + [1],
                                         padding="SAME")
                    return net
                net = tf.map_fn(lambda x:fun(x[0],x[1]),elems=(bboxes,batch_index),dtype=tf.float32)
            else:
                batch_index = tf.reshape(batch_index,[-1])
                bboxes = tf.reshape(bboxes,[-1,4])
                if self.multiplier is not None:
                    bboxes = odb.scale_bboxes(bboxes,scale=self.multiplier)
                net = tf.image.crop_and_resize(image=fmap,boxes=bboxes,box_ind=batch_index,crop_size=[height,width])
                net = tf.nn.max_pool(net,ksize=[1]+self.bin_size+[1],strides=[1]+self.bin_size+[1],padding="SAME")'''
            batch_index = tf.reshape(batch_index, [-1])
            bboxes = tf.reshape(bboxes, [-1, 4])
            if self.multiplier is not None and len(self.multiplier)>=2:
                bboxes = odb.scale_bboxes(bboxes, scale=self.multiplier)
            net = tf.image.crop_and_resize(image=fmap, boxes=bboxes, box_ind=batch_index, crop_size=[height, width])
            net = tf.nn.max_pool(net, ksize=[1] + self.bin_size + [1], strides=[1] + self.bin_size + [1],
                                 padding="SAME")
            return net

DFROI=WROIAlign
'''
def boxesNMS(bboxes,labels,probs,threshold=0.5,classes_wise=True):
    bboxes,labels,indices = boxes_nms(bboxes,labels,threshold=threshold,classes_wise=classes_wise)
    probs = tf.gather(probs,indices)
    return bboxes,labels,probs
'''

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