#coding=utf-8
import tensorflow as tf
from object_detection2.modeling.matcher import Matcher
from object_detection2.standard_names import *
from wmodule import WModule
from basic_tftools import batch_gather


def get_img_size_from_batched_inputs(inputs):
    image = inputs[IMAGE]
    return tf.shape(image)[1:3]

class RemoveSpecifiedResults(object):
    def __init__(self,pred_fn=None):
       self.matcher = Matcher(thresholds=[1e-4],parent=self) 
       self.pred_fn = pred_fn if pred_fn is not None else self.__pred_fn
        
        
    def __call__(self, gbboxes,glabels,bboxes,labels):
        '''
        
        :param gbboxes: [1,X,4]
        :param glabels: [1,X]
        :param bboxes: [1,Y,4]
        :param labels: [1,Y]
        :return: 
        '''
        glength = tf.reshape(tf.shape(glabels)[1],[1])
        removed_gbboxes = self.pred_fn(gbboxes,glabels)
        labels, scores, indices = self.matcher(bboxes,gbboxes,glabels,glength)
        is_bboxes_removed = tf.batch_gather(removed_gbboxes,tf.nn.relu(indices))
        is_bboxes_removed = tf.logical_or(is_bboxes_removed,tf.less(indices,0))
        gbboxes = tf.expand_dims(tf.boolean_mask(gbboxes,removed_gbboxes),0)
        glabels = tf.expand_dims(tf.boolean_mask(glabels,removed_gbboxes),0)
        bboxes = tf.expand_dims(bboxes,is_bboxes_removed)
        labels = tf.expand_dims(labels,is_bboxes_removed)
        return gbboxes,glabels,bboxes,labels,removed_gbboxes,is_bboxes_removed

    @staticmethod
    def __pred_fn(gbboxes,glabels):
        return tf.zeros_like(glabels,dtype=tf.bool)


def replace_with_gtlabels(bboxes,labels,length,gtbboxes,gtlabels,gtlength,threshold=0.5):
    parent = WModule(cfg=None)
    matcher = Matcher(
        thresholds=[threshold],
        allow_low_quality_matches=False,
        cfg=None,
        parent=parent,
    )
    n_labels,_,_ = matcher(bboxes,gtbboxes,gtlabels,gtlength)
    labels = tf.where(tf.greater(n_labels,0),n_labels,labels)
    return labels

def replace_with_gtbboxes(bboxes,labels,length,gtbboxes,gtlabels,gtlength,threshold=0.5):
    parent = WModule(cfg=None)
    matcher = Matcher(
        thresholds=[threshold],
        allow_low_quality_matches=False,
        cfg=None,
        parent=parent,
    )
    n_labels,_,indices = matcher(bboxes,gtbboxes,gtlabels,gtlength)
    n_labels = tf.expand_dims(n_labels,axis=-1)
    n_labels = tf.tile(n_labels,[1,1,4])
    n_bboxes = batch_gather(gtbboxes,tf.nn.relu(indices))
    bboxes = tf.where(tf.greater(n_labels,0),n_bboxes,bboxes)
    return bboxes
