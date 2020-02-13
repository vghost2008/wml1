#coding=utf-8
import wtfop.wtfop_ops as wop
import wmodule
import tensorflow as tf

class Matcher(wmodule.WChildModule):
    def __init__(self,thresholds,allow_low_quality_matches=False,same_pos_label=None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if len(thresholds) == 1:
            thresholds = [thresholds[0],thresholds[0]]
        self.thresholds = thresholds
        self.allow_low_quality_matches = allow_low_quality_matches
        self.same_pos_label = same_pos_label

    def forward(self,boxes,gboxes,glabels,glength):
        labels,scores,indices = wop.matcher(bboxes=boxes,gboxes=gboxes,
                           glabels=glabels,
                           length=glength,
                           neg_threshold=self.thresholds[0],
                           pos_threshold=self.thresholds[1],
                           max_overlap_as_pos=self.allow_low_quality_matches)

        if self.same_pos_label:
            labels = tf.where(tf.greater(labels,0),tf.ones_like(labels)*self.same_pos_label,labels)

        return labels,scores,indices
