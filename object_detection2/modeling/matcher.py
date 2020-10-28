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
        print("Matcher")
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

    @staticmethod
    def moments(data,threshold,axes=-1):
        mask = tf.greater_equal(data,threshold)
        mask_f = tf.cast(mask,data.dtype)
        data_f = tf.where(mask,data,tf.zeros_like(data))
        data_sum = tf.reduce_sum(data_f,axis=axes,keepdims=True)
        data_nr = tf.maximum(tf.reduce_sum(mask_f,axis=axes,keepdims=True),1)
        data_mean = data_sum/data_nr
        s_diff = tf.squared_difference(data, tf.stop_gradient(data_mean))
        s_diff = tf.where(mask,s_diff,tf.zeros_like(s_diff))
        variance = tf.reduce_sum(
            s_diff,
            axis=axes,
            keepdims=True,
            name="variance")/data_nr
        return data_mean,variance

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
            assert isinstance(boxes_len,(list,tuple)), "error boxes len type."
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
            offsets = [0]
            with tf.name_scope("get_offset"):
                for i in range(len(boxes_len)-1):
                    n_off = offsets[-1]+boxes_len[i]
                    offsets.append(n_off)
            pos_indices = []
            for tl,bl,dism in zip(offsets,boxes_len,dis_matrix):
                values,indices = tf.nn.top_k(-dism,k=tf.minimum(self.k,bl),sorted=False)
                indices = indices+tl
                pos_indices.append(indices)

            pos_indices = tf.concat(pos_indices,axis=-1)
            pos_ious = btf.batch_gather(iou_matrix,pos_indices,name="gather_pos_ious")
            iou_mean,iou_var = self.moments(pos_ious,threshold=self.MIN_IOU_THRESHOLD,axes=[-1])
            #wsummary.histogram_or_scalar(iou_mean,"iou_mean")
            with tf.device("/cpu:0"):
                max_iou_threshold = tf.reduce_max(pos_ious,axis=-1,keepdims=True)
                iou_std = tf.sqrt(iou_var)
                iou_threshold = iou_mean+iou_std
                iou_threshold = tf.minimum(max_iou_threshold,iou_threshold)
                '''
                原算法中表示的为仅从上面的topk中取正样本，这里从所有的样本中取正样本
                '''
                is_pos = tf.logical_and(iou_matrix>=iou_threshold,is_center_in_gtboxes)
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

@MATCHER.register()
class ATSSMatcher3(wmodule.WChildModule):
    MIN_IOU_THRESHOLD = 0.1
    def __init__(self,thresholds,same_pos_label=None,*args,**kwargs):
        '''
        '''
        super().__init__(*args,**kwargs)
        self.same_pos_label = same_pos_label
        self.thresholds = thresholds

        print(f"ATSSMatcher v3.0, thresholds={self.thresholds}")

    @wmlt.add_name_scope
    def get_threshold(self,iou_matrix):
        '''
        iou_matrix: [B,GT_nr,Anchor_nr]
        X = GT_nr, Y=Anchor_nr
        return:
        [B,GT]
        '''
        B,X,Y = btf.combined_static_and_dynamic_shape(iou_matrix)
        iou_matrix = tf.reshape(iou_matrix,[B*X,Y])
        def fn(ious):
            mask = tf.greater(ious,self.MIN_IOU_THRESHOLD)
            def fn0():
                p_ious = tf.boolean_mask(ious,mask)
                mean,var = tf.nn.moments(p_ious,axes=-1)
                std = tf.sqrt(var)
                return mean+std
            def fn1():
                return tf.constant(1.0,dtype=tf.float32)
            return tf.cond(tf.reduce_any(mask),fn0,fn1)
        threshold = tf.map_fn(fn,elems=iou_matrix,back_prop=False)
        threshold = tf.reshape(threshold,[B,X])
        return tf.stop_gradient(threshold)

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
        with tf.name_scope("ATTSMatcher3"):
            iou_matrix = odb.batch_bboxes_pair_wrapv2(gboxes,boxes,
                                                      fn=odb.get_iou_matrix,
                                                      len0=glength,
                                                      scope="get_iou_matrix")
            is_center_in_gtboxes = odb.batch_bboxes_pair_wrapv2(gboxes,boxes,
                                                                fn=odb.is_center_in_boxes,
                                                                len0=glength,
                                                                dtype=tf.bool,
                                                                scope="get_is_center_in_gtbboxes")
            wsummary.variable_summaries_v2(iou_matrix,"iou_matrix")

            with tf.device("/cpu:0"):
                iou_threshold = self.get_threshold(iou_matrix)
                iou_threshold = tf.minimum(iou_threshold,self.thresholds[-1])
                iou_matrix = tf.where(is_center_in_gtboxes,iou_matrix,tf.zeros_like(iou_matrix))
                scores,index = tf.nn.top_k(tf.transpose(iou_matrix,perm=[0,2,1]),k=1)
                B,Y,_ = btf.combined_static_and_dynamic_shape(gboxes)
                index = tf.squeeze(index,axis=-1)
                scores = tf.squeeze(scores,axis=-1)
                threshold = wmlt.batch_gather(iou_threshold,index)
                labels = wmlt.batch_gather(glabels,index,name="gather_labels",
                                           parallel_iterations=B,
                                           back_prop=False)
                is_good_score = tf.greater(scores,self.MIN_IOU_THRESHOLD)
                is_good_score = tf.logical_and(is_good_score,scores>=threshold)
                labels = tf.where(is_good_score,labels,tf.zeros_like(labels))
                margin = self.thresholds[-1]-self.thresholds[0]
                is_in_mid_threshold = tf.logical_and(scores<threshold,scores>threshold-margin)
                is_ignore = tf.logical_and(is_in_mid_threshold,scores>self.MIN_IOU_THRESHOLD+margin)
                labels = tf.where(is_ignore,tf.ones_like(labels)*-1,labels)
                index = tf.where(is_good_score,index,tf.ones_like(index)*-1)

            if self.same_pos_label:
                labels = tf.where(tf.greater(labels, 0), tf.ones_like(labels) * self.same_pos_label, labels)

            return tf.stop_gradient(labels),tf.stop_gradient(scores),tf.stop_gradient(index)

@MATCHER.register()
class ATSSMatcher4(wmodule.WChildModule):
    '''
    相比于ATSSMatcher3, ATSSMatcher4不会处理threshold[0]与threshold[1]之间的这部分样本
    '''
    MIN_IOU_THRESHOLD = 0.1
    def __init__(self,thresholds,same_pos_label=None,*args,**kwargs):
        '''
        '''
        super().__init__(*args,**kwargs)
        self.same_pos_label = same_pos_label
        self.thresholds = thresholds

        print(f"ATSSMatcher v4.0, thresholds={self.thresholds}")

    @wmlt.add_name_scope
    def get_threshold(self,iou_matrix):
        '''
        iou_matrix: [B,GT_nr,Anchor_nr]
        X = GT_nr, Y=Anchor_nr
        return:
        [B,GT]
        '''
        B,X,Y = btf.combined_static_and_dynamic_shape(iou_matrix)
        iou_matrix = tf.reshape(iou_matrix,[B*X,Y])
        def fn(ious):
            mask = tf.greater(ious,self.MIN_IOU_THRESHOLD)
            def fn0():
                p_ious = tf.boolean_mask(ious,mask)
                mean,var = tf.nn.moments(p_ious,axes=-1)
                std = tf.sqrt(var)
                return mean+std
            def fn1():
                return tf.constant(1.0,dtype=tf.float32)
            return tf.cond(tf.reduce_any(mask),fn0,fn1)
        threshold = tf.map_fn(fn,elems=iou_matrix,back_prop=False)
        threshold = tf.reshape(threshold,[B,X])
        return tf.stop_gradient(threshold)

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
        with tf.name_scope("ATTSMatcher4"):
            iou_matrix = odb.batch_bboxes_pair_wrapv2(gboxes,boxes,
                                                      fn=odb.get_iou_matrix,
                                                      len0=glength,
                                                      scope="get_iou_matrix")
            is_center_in_gtboxes = odb.batch_bboxes_pair_wrapv2(gboxes,boxes,
                                                                fn=odb.is_center_in_boxes,
                                                                len0=glength,
                                                                dtype=tf.bool,
                                                                scope="get_is_center_in_gtbboxes")
            wsummary.variable_summaries_v2(iou_matrix,"iou_matrix")

            with tf.device("/cpu:0"):
                iou_threshold = self.get_threshold(iou_matrix)
                iou_threshold = tf.minimum(iou_threshold,self.thresholds[-1])
                iou_matrix = tf.where(is_center_in_gtboxes,iou_matrix,tf.zeros_like(iou_matrix))
                scores,index = tf.nn.top_k(tf.transpose(iou_matrix,perm=[0,2,1]),k=1)
                B,Y,_ = btf.combined_static_and_dynamic_shape(gboxes)
                index = tf.squeeze(index,axis=-1)
                scores = tf.squeeze(scores,axis=-1)
                threshold = wmlt.batch_gather(iou_threshold,index)
                labels = wmlt.batch_gather(glabels,index,name="gather_labels",
                                           parallel_iterations=B,
                                           back_prop=False)
                is_good_score = tf.greater(scores,self.MIN_IOU_THRESHOLD)
                is_good_score = tf.logical_and(is_good_score,scores>=threshold)
                labels = tf.where(is_good_score,labels,tf.zeros_like(labels))
                index = tf.where(is_good_score,index,tf.ones_like(index)*-1)

            if self.same_pos_label:
                labels = tf.where(tf.greater(labels, 0), tf.ones_like(labels) * self.same_pos_label, labels)

            return tf.stop_gradient(labels),tf.stop_gradient(scores),tf.stop_gradient(index)

@MATCHER.register()
class DynamicMatcher(wmodule.WChildModule):
    MIN_IOU_THRESHOLD = 0.1
    def __init__(self,thresholds=[0.0],same_pos_label=None,*args,**kwargs):
        '''
        '''
        super().__init__(*args,**kwargs)
        self.same_pos_label = same_pos_label
        self.thresholds = thresholds

        print(f"DynamicMatcher v1.0, thresholds={self.thresholds}")

    @staticmethod
    def moments(data,weights,threshold,axes=-1):
        mask = tf.greater_equal(data,threshold)
        mask_f = tf.cast(mask,data.dtype)
        if weights.dtype != data.dtype:
            weights = tf.cast(weights,data.dtype)
        mask_wf = mask_f*weights
        data_f = tf.where(mask,data,tf.zeros_like(data))
        data_wf = data_f*weights
        data_sum = tf.reduce_sum(data_wf,axis=axes,keepdims=True)
        data_nr = tf.maximum(tf.reduce_sum(mask_wf,axis=axes,keepdims=True),1)
        data_mean = data_sum/data_nr
        s_diff = tf.squared_difference(data, tf.stop_gradient(data_mean))
        s_diff = tf.where(mask,s_diff,tf.zeros_like(s_diff))*weights
        variance = tf.reduce_sum(
            s_diff,
            axis=axes,
            keepdims=True,
            name="variance")/data_nr
        return data_mean,variance

    @wmlt.add_name_scope
    def get_threshold(self,iou_matrix,anchor_weights):
        '''
        iou_matrix: [B,GT_nr,Anchor_nr]
        X = GT_nr, Y=Anchor_nr
        return:
        [B,GT]
        '''
        B,GT_nr,Anchor_nr = wmlt.combined_static_and_dynamic_shape(iou_matrix)
        anchor_weights = tf.reshape(anchor_weights,[1,1,Anchor_nr])
        iou_mean, iou_var = self.moments(iou_matrix, weights=anchor_weights,
                                         threshold=self.MIN_IOU_THRESHOLD, axes=[-1])
        iou_std = tf.sqrt(iou_var)
        iou_threshold = iou_mean + iou_std
        iou_threshold = tf.squeeze(iou_threshold,axis=-1)

        return tf.stop_gradient(iou_threshold)

    @wmlt.add_name_scope
    def get_anchor_weights(self,boxes_len):

        boxes_len_f = [tf.to_float(x) for x in boxes_len]
        scales = [tf.to_float(boxes_len[0])/x for x in boxes_len_f]
        weights = []
        for s,l in zip(scales,boxes_len):
            w = tf.ones(shape=[l],dtype=tf.float32)*s
            weights.append(w)

        return tf.concat(weights,axis=-1)


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
        with tf.name_scope("DynamicMatcher"):
            assert isinstance(boxes_len,(list,tuple)), "error boxes len type."
            iou_matrix = odb.batch_bboxes_pair_wrapv2(gboxes,boxes,
                                                      fn=odb.get_iou_matrix,
                                                      len0=glength,
                                                      scope="get_iou_matrix")
            is_center_in_gtboxes = odb.batch_bboxes_pair_wrapv2(gboxes,boxes,
                                                                fn=odb.is_center_in_boxes,
                                                                len0=glength,
                                                                dtype=tf.bool,
                                                                scope="get_is_center_in_gtbboxes")
            wsummary.variable_summaries_v2(iou_matrix,"iou_matrix")

            with tf.device("/cpu:0"):
                anchor_weights = self.get_anchor_weights(boxes_len)
                iou_threshold = self.get_threshold(iou_matrix,anchor_weights)
                if self.thresholds[-1]>self.MIN_IOU_THRESHOLD:
                    print(f"DynamicMatcher use thresholds ceiling {self.thresholds[-1]}.")
                    iou_threshold = tf.minimum(iou_threshold,self.thresholds[-1])
                iou_matrix = tf.where(is_center_in_gtboxes,iou_matrix,tf.zeros_like(iou_matrix))
                scores,index = tf.nn.top_k(tf.transpose(iou_matrix,perm=[0,2,1]),k=1)
                B,Y,_ = btf.combined_static_and_dynamic_shape(gboxes)
                index = tf.squeeze(index,axis=-1)
                scores = tf.squeeze(scores,axis=-1)
                threshold = wmlt.batch_gather(iou_threshold,index)
                labels = wmlt.batch_gather(glabels,index,name="gather_labels",
                                           parallel_iterations=B,
                                           back_prop=False)
                is_good_score = tf.greater(scores,self.MIN_IOU_THRESHOLD)
                is_good_score = tf.logical_and(is_good_score,scores>=threshold)
                labels = tf.where(is_good_score,labels,tf.zeros_like(labels))
                index = tf.where(is_good_score,index,tf.ones_like(index)*-1)

            if self.same_pos_label:
                labels = tf.where(tf.greater(labels, 0), tf.ones_like(labels) * self.same_pos_label, labels)

            return tf.stop_gradient(labels),tf.stop_gradient(scores),tf.stop_gradient(index)

@MATCHER.register()
class MatcherV2(wmodule.WChildModule):
    def __init__(self,thresholds,same_pos_label=None,*args,**kwargs):
        '''
        :param thresholds: [threshold] or [threshold_low,threshold_high]
        :param allow_low_quality_matches: if it's true, the box which match some gt box best will be set to positive
        :param same_pos_label: int, if it's not None, then all positive boxes' label will be set to same_pos_label
        '''
        super().__init__(*args,**kwargs)
        if len(thresholds) == 1:
            thresholds = [thresholds[0],thresholds[0]]
        print(f"MatcherV2, thresholds={thresholds}")
        self.thresholds = thresholds
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
        labels,scores,indices = wop.matcherv2(bboxes=boxes,gboxes=gboxes,
                                            glabels=glabels,
                                            length=glength,
                                            threshold=self.thresholds)

        if self.same_pos_label:
            labels = tf.where(tf.greater(labels,0),tf.ones_like(labels)*self.same_pos_label,labels)

        return labels,scores,indices
