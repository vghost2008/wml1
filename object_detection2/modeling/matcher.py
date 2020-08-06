#coding=utf-8
import wtfop.wtfop_ops as wop
import wmodule
import tensorflow as tf
import basic_tftools as btf
from .build_matcher import MATCHER
import wml_tfutils as wmlt
import object_detection2.bboxes as odb
import wsummary

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
    def forward(self,boxes,gboxes,glabels,glength,*args,**kwargs):
        '''
        :param boxes: [1,X,4] or [batch_size,X,4] proposal boxes
        :param gboxes: [batch_size,Y,4] groundtruth boxes
        :param glabels: [batch_size,Y] groundtruth labels
        :param glength: [batch_size] boxes size
        :return:
        labels: [batch_size,X,4], the label of boxes, -1 indict ignored box, which will not calculate loss,
        0 is background
        scores: [batch_size,X], the overlap score with boxes' match gt box
        indices: [batch_size,X] the index of matched gt boxes when it's a positive anchor box, else it's -1
        '''
        labels,scores,indices = wop.matcher(bboxes=boxes,gboxes=gboxes,
                           glabels=glabels,
                           length=glength,
                           neg_threshold=self.thresholds[0],
                           pos_threshold=self.thresholds[1],
                           max_overlap_as_pos=self.allow_low_quality_matches,
                           force_in_gtbox=False)

        if self.same_pos_label:
            labels = tf.where(tf.greater(labels,0),tf.ones_like(labels)*self.same_pos_label,labels)

        return labels,scores,indices

'''
Bridging the Gap Between Anchor-based and Anchor-free Detection via
Adaptive Training Sample Selection
'''
@MATCHER.register()
class ATSSMatcher(wmodule.WChildModule):
    MIN_IOU_THRESHOLD = 0.1
    def __init__(self,k=9,same_pos_label=None,*args,**kwargs):
        '''
        '''
        super().__init__(*args,**kwargs)
        self.k = k
        self.same_pos_label = same_pos_label

        print(f"ATSSMatcher v1.0 k={k}")

    def forward(self,boxes,gboxes,glabels,glength,boxes_len,*args,**kwargs):
        '''
        :param boxes: [1,X,4] or [batch_size,X,4] proposal boxes
        :param gboxes: [batch_size,Y,4] groundtruth boxes
        :param glabels: [batch_size,Y] groundtruth labels
        :param glength: [batch_size] boxes size
        :param boxes_len: [len0,len1,len2,...] sum(boxes_len)=X, boxes len in each layer
        :return:
        labels: [batch_size,X,4], the label of boxes, -1 indict ignored box, which will not calculate loss,
        0 is background
        scores: [batch_size,X], the overlap score with boxes' match gt box
        indices: [batch_size,X] the index of matched gt boxes when it's a positive anchor box, else it's -1
        '''
        with tf.name_scope("ATTSMatcher"):
            dis_matrix = odb.batch_bboxes_pair_wrapv2(gboxes,boxes,
                                                    fn=odb.get_bboxes_dis,
                                                    len0=glength,
                                                    scope="get_dis_matrix")
            iou_matrix = odb.batch_bboxes_pair_wrapv2(gboxes,boxes,
                                                    fn=odb.get_iou_matrix,
                                                    len0=glength,
                                                    scope="get_iou_matrix")
            is_center_in_gtboxes = odb.batch_bboxes_pair_wrapv2(gboxes,boxes,
                                                    fn=odb.is_center_in_boxes,
                                                    len0=glength,
                                                    dtype=tf.bool,
                                                    scope="get_is_center_in_gtbboxes")
            #dis_matrix = tf.Print(dis_matrix,[tf.shape(dis_matrix),tf.reduce_sum(boxes_len)],summarize=100)
            dis_matrix = tf.split(dis_matrix,boxes_len,axis=2)
            pos_indices = []
            for tl,dism in zip([0]+boxes_len[:-1],dis_matrix):
                values,indices = tf.nn.top_k(-dism,k=self.k,sorted=False)
                indices = indices+tl
                pos_indices.append(indices)

            pos_indices = tf.concat(pos_indices,axis=-1)
            pos_ious = btf.batch_gather(iou_matrix,pos_indices,name="gather_pos_ious")
            iou_mean,iou_var = tf.nn.moments(pos_ious,keep_dims=True,axes=[-1])
            #wsummary.histogram_or_scalar(iou_mean,"iou_mean")
            with tf.device("/cpu:0"):
                iou_std = tf.sqrt(iou_var)
                iou_threshold = iou_mean+iou_std
                #wsummary.histogram_or_scalar(iou_std, "iou_std")
                #wsummary.histogram_or_scalar(iou_threshold, "iou_threshold")
                '''
                原算法中表示的为仅从上面的topk中取正样本，这里从所有的样本中取正样本
                '''
                is_pos = tf.logical_and(iou_matrix>iou_threshold,is_center_in_gtboxes)
                iou_matrix = tf.where(is_pos,iou_matrix,tf.zeros_like(iou_matrix))
                scores,index = tf.nn.top_k(tf.transpose(iou_matrix,perm=[0,2,1]),k=1)
                B,Y,_ = btf.combined_static_and_dynamic_shape(gboxes)
                index = tf.squeeze(index,axis=-1)
                scores = tf.squeeze(scores,axis=-1)
                labels = wmlt.batch_gather(glabels,index,name="gather_labels",
                                           parallel_iterations=B,
                                           back_prop=False)
                is_good_score = tf.greater(scores,self.MIN_IOU_THRESHOLD)
                labels = tf.where(is_good_score,labels,tf.zeros_like(labels))
                index = tf.where(is_good_score,index,tf.ones_like(index)*-1)

                #iou_matrix=iou_matrix[:1,:glength[0]]
                #iou_matrix = tf.reduce_sum(iou_matrix,axis=-1)
                #wsummary.histogram_or_scalar(iou_matrix,"iou_matrix")

            if self.same_pos_label:
                labels = tf.where(tf.greater(labels, 0), tf.ones_like(labels) * self.same_pos_label, labels)
            return tf.stop_gradient(labels),tf.stop_gradient(scores),tf.stop_gradient(index)
