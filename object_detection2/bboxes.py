#coding=utf-8
import tensorflow as tf
import wtfop.wtfop_ops as wtfop
import wml_tfutils as wmlt

def random_distored_boxes(bboxes,limits=[0.,0.,0.],size=1,keep_org=True):
    if bboxes.get_shape().ndims == 2:
        return wtfop.random_distored_boxes(boxes=bboxes,limits=limits,size=size,keep_org=keep_org)
    else:
        return tf.map_fn(lambda x: wtfop.random_distored_boxes(boxes=x,limits=limits,size=size,keep_org=keep_org),
                         elems=bboxes,back_prop=False)

'''
bboxes:[batch_size,X,4] or [X,4]
return:
[batch_size,X]
'''
def box_area(bboxes):
    with tf.name_scope("box_area"):
        ymin,xmin,ymax,xmax = tf.unstack(bboxes,axis=len(bboxes.get_shape())-1)
        h = ymax-ymin
        w = xmax-xmin
        return tf.multiply(h,w)

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
            boxes = tf.pad(boxes,[[0,k-lens],[0,0]])
            labels = tf.pad(labels,[[0,k-lens]])
            if sort:
                nms_indexs = tf.gather(r_index,nms_indexs)
            nms_indexs = tf.pad(nms_indexs,[[0,k-lens]])
            return [boxes,labels,nms_indexs,r_len]
        boxes,labels,nms_indexs,lens = wmlt.static_or_dynamic_map_fn(lambda x:do_nms(x[0],x[1],x[2],x[3]),elems=[bboxes,classes,confidence,lens],
                                                                     dtype=(tf.float32,tf.int32,tf.int32,tf.int32))
        return boxes,labels,nms_indexs,lens