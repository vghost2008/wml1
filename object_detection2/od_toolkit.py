#coding=utf-8
import tensorflow as tf
import wtfop.wtfop_ops as wop
import wml_tfutils as wmlt

'''
boxes:[N,4]
'''
def random_select_boxes(boxes,size,len=None):
    with tf.name_scope("random_select_boxes"):
        if len is None:
            data_nr = tf.shape(boxes)[0]
        else:
            data_nr = len
        indexs = tf.range(data_nr)
        indexs = wop.wpad(indexs, [0, tf.reshape(size - data_nr,())])
        indexs = tf.random_shuffle(indexs)
        indexs = tf.random_crop(indexs, [size])
        boxes = tf.gather(boxes, indexs)
        return boxes, indexs

'''
boxes:[batch_size,N,4]
lens:[batch_size]
'''
def batched_random_select_boxes(boxes,lens,size):
    with tf.name_scope("random_select_boxes"):
        boxes,indexs = tf.map_fn(lambda x:random_select_boxes(x[0],size,x[1]),elems=(boxes,lens),dtype=(tf.float32,tf.int32))
        batch_size = boxes.get_shape().as_list()[0]
        boxes = wmlt.reshape(boxes,[batch_size,size,4])
        indexs = wmlt.reshape(indexs,[batch_size,size])
        return boxes,indexs