#coding=utf-8
import tensorflow as tf
import wtfop.wtfop_ops as wtfop

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