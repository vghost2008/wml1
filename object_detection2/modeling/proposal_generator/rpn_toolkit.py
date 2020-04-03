#coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
import wtfop.wtfop_ops as wop


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
            anchors_num_per_level,
    ):
    if len(anchors_num_per_level) == 1:
        return find_top_rpn_proposals_for_single_level(proposals,pred_objectness_logits,nms_thresh,pre_nms_topk,
                                                       post_nms_topk)
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
                                                              post_nms_topk=post_nms_topk//(3**i))
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
):
    with tf.name_scope("find_top_rpn_proposals_for_single_level"):
        class_prediction = pred_objectness_logits
        probability = class_prediction

        '''
        通过top_k+gather排序
        In Detectron2, they chosen the top candiate_nr*6 boxes
        '''
        probability,indices = tf.nn.top_k(probability,k=tf.minimum(pre_nms_topk,tf.shape(probability)[1]))
        proposals = wmlt.batch_gather(proposals,indices)


        def fn(bboxes,probability):
            labels = tf.ones(tf.shape(bboxes)[0],dtype=tf.int32)
            boxes,labels,indices = wop.boxes_nms_nr2(bboxes,labels,k=post_nms_topk,threshold=nms_thresh,confidence=probability)
            probability = tf.gather(probability,indices)
            return boxes,probability

        boxes,probability = tf.map_fn(lambda x:fn(x[0],x[1]),elems=(proposals,probability),
                                      dtype=(tf.float32,tf.float32),back_prop=False)
        return tf.stop_gradient(boxes),tf.stop_gradient(probability)
