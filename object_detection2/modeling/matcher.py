#coding=utf-8
import wtfop.wtfop_ops as wop
import wmodule
import tensorflow as tf
import basic_tftools as btf
from .build_matcher import MATCHER

@MATCHER.register()
class Matcher(wmodule.WChildModule):
    def __init__(self,thresholds,allow_low_quality_matches=False,same_pos_label=None,*args,**kwargs):
        '''
        :param thresholds: [threshold] or [threshold_low,threshold_high]
        :param allow_low_quality_matches: if it's true, the box which match some gt box best will be set to positive
        :param same_pos_label: int, if it's not None, then all positive boxes' label will be set to same_pos_label
        '''
        super().__init__(*args,**kwargs)
        if len(thresholds) == 1:
            thresholds = [thresholds[0],thresholds[0]]
        self.thresholds = thresholds
        self.allow_low_quality_matches = allow_low_quality_matches
        self.same_pos_label = same_pos_label

    @btf.show_input_shape
    def forward(self,boxes,gboxes,glabels,glength):
        '''
        :param boxes: [1,X,4] or [batch_size,X,4] proposal boxes
        :param gboxes: [batch_size,Y,4] groundtruth boxes
        :param glabels: [batch_size,Y] groundtruth labels
        :param glength: [batch_size] boxes size
        :return:
        labels: [batch_size,X,4], the label of boxes, -1 indict ignored box
        scores: [batch_size,X], the overlap score with boxes' match gt box
        indices: [batch_size,X] the index of matched gt boxes
        '''
        labels,scores,indices = wop.matcher(bboxes=boxes,gboxes=gboxes,
                           glabels=glabels,
                           length=glength,
                           neg_threshold=self.thresholds[0],
                           pos_threshold=self.thresholds[1],
                           max_overlap_as_pos=self.allow_low_quality_matches)

        if self.same_pos_label:
            labels = tf.where(tf.greater(labels,0),tf.ones_like(labels)*self.same_pos_label,labels)

        return labels,scores,indices
