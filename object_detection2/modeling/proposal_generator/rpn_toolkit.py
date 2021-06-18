#coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
import tfop
import basic_tftools as btf


'''
this function get exactly post_nms_topk boxes by the heristic method
firt remove boxes by nums, and then get the top post_nms_topk boxes, if there not enough boxes after nms,
the boxes in the front will be add to result.

pred_objectness_logits:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率或正相关的指标(logits)
shape为[batch_size,X]
proposals:模型预测的候选框, shape为[batch_size,X,4]
nms_thresh:nms阀值
pre_nms_topk: poposals的数量先降为pre_nms_topk后再进行NMS处理(NMS处理比较耗时)
post_nms_topk: NMS处理后剩余的proposal数量
pre_nms_topk/post都仅针对分辨率最高的feature map, 附后的feature map相关值会减小
anchors_nms_per_level:每一层的anchors数量, list of Tensor
In Detectron2, they separaty select proposals from each level, use the same pre_nms_topk and post_nams_topk, in order to
simple arguments settin, we use the follow arguemnts[pre_nms_topk,pos_nms_topk] for level1, [pre_nms_topk//4,pos_nms_topk//4]
for level2, and so on.

返回:
boxes:[Y,4]
labels:[Y]
probability:[Y]
'''
def find_top_rpn_proposals(
            proposals,
            pred_objectness_logits,
            nms_thresh,
            pre_nms_topk,
            post_nms_topk,
            anchors_num_per_level=None,
            score_threshold=None,
            is_training=True,
            pre_nms_topk_max_per_layer=-1,
    ):
    if anchors_num_per_level is None or len(anchors_num_per_level) == 1:
        return find_top_rpn_proposals_for_single_level(proposals,pred_objectness_logits,nms_thresh,pre_nms_topk,
                                                       post_nms_topk,
                                                       score_threshold=score_threshold,
                                                       is_training=is_training,
                                                       pre_nms_topk_max_per_layer=pre_nms_topk_max_per_layer)
    with tf.name_scope("find_top_rpn_proposals"):
        proposals = tf.split(proposals,num_or_size_splits=anchors_num_per_level,axis=1)
        pred_objectness_logits = tf.split(pred_objectness_logits,num_or_size_splits=anchors_num_per_level,axis=1)
        boxes = []
        probabilitys = []
        for i in range(len(anchors_num_per_level)):
            t_boxes,t_probability = find_top_rpn_proposals_for_single_level(proposals=proposals[i],
                                                              pred_objectness_logits=pred_objectness_logits[i],
                                                              nms_thresh=nms_thresh,
                                                              pre_nms_topk=pre_nms_topk//(3**i),
                                                              post_nms_topk=post_nms_topk//(3**i),
                                                              score_threshold=score_threshold,
                                                              is_training=is_training,
                                                              pre_nms_topk_max_per_layer=pre_nms_topk_max_per_layer)
            boxes.append(t_boxes)
            probabilitys.append(t_probability)
        return tf.concat(boxes,axis=1),tf.concat(probabilitys,axis=1)
'''
this function get exactly candiate_nr boxes by the heristic method
firt remove boxes by nums, and then get the top candiate_nr boxes, if there not enough boxes after nms,
the boxes in the front will be add to result.

pred_objectness_logits:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率或正相关的指标(logits)
shape为[batch_size,X]
proposals:模型预测的候选框, shape为[batch_size,X,4]
nms_thresh:nms阀值
pre_nms_topk: poposals的数量先降为pre_nms_topk后再进行NMS处理(NMS处理比较耗时)
post_nms_topk: NMS处理后剩余的proposal数量
pre_nms_topk/post都仅针对分辨率最高的feature map, 附后的feature map相关值会减小
anchors_nms_per_level:每一层的anchors数量, list of Tensor

返回:
boxes:[Y,4]
labels:[Y]
probability:[Y]
'''
def find_top_rpn_proposals_for_single_level(
        proposals,
        pred_objectness_logits,
        nms_thresh,
        pre_nms_topk,
        post_nms_topk,
        score_threshold=-1.0,
        is_training=True,
        pre_nms_topk_max_per_layer=-1,
):
    with tf.name_scope("find_top_rpn_proposals_for_single_level"):
        '''
        通过top_k+gather排序
        In Detectron2, they chosen the top candiate_nr*6 boxes
        '''
        if pre_nms_topk_max_per_layer>10:
            topk_nr = tf.minimum(pre_nms_topk,tf.shape(pred_objectness_logits)[1])
            print(f"pre_nms_topk_max_per_layer = {pre_nms_topk_max_per_layer}.")
            topk_nr = tf.minimum(topk_nr,pre_nms_topk_max_per_layer)
        else:
            topk_nr = tf.minimum(pre_nms_topk, tf.shape(pred_objectness_logits)[1])
        probability,indices = tf.nn.top_k(pred_objectness_logits,
                                          k=topk_nr)
        proposals = wmlt.batch_gather(proposals,indices)
        batch_size = pred_objectness_logits.get_shape().as_list()[0]
        if not is_training and batch_size>1:
            print("RPN: Inference on multi images.")

        def fn(bboxes,probability):
            labels = tf.ones(tf.shape(bboxes)[0],dtype=tf.int32)
            if is_training or batch_size>1:
                boxes,labels,indices = tfop.boxes_nms_nr2(bboxes,labels,k=post_nms_topk,threshold=nms_thresh,confidence=probability)
                probability = tf.gather(probability,indices)
            else:
                boxes,labels,indices = tfop.boxes_nms(bboxes,labels,k=post_nms_topk,threshold=nms_thresh,confidence=probability)
                probability = tf.gather(probability,indices)
                if score_threshold > 1e-10:
                    p_mask = tf.greater(probability,score_threshold)

                    indices = tf.constant([[0]],dtype=tf.int32)
                    updates = tf.constant([1],dtype=tf.int32)
                    shape = tf.shape(p_mask)
                    lp_mask= tf.cast(tf.scatter_nd(indices, updates, shape),tf.bool)
                    p_mask = tf.logical_or(p_mask,lp_mask)

                    probability = tf.boolean_mask(probability,p_mask)
                    boxes = tf.boolean_mask(boxes,p_mask)

            return [boxes,probability]

        boxes,probability = btf.try_static_or_dynamic_map_fn(lambda x:fn(x[0],x[1]),elems=[proposals,probability],
                                      dtype=[tf.float32,tf.float32],back_prop=False)
        
        return tf.stop_gradient(boxes),tf.stop_gradient(probability)
