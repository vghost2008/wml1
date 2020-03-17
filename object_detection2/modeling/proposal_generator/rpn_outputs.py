import tensorflow as tf
import wml_tfutils as wmlt
import wtfop.wtfop_ops as wop
from object_detection2.config.config import global_cfg
import itertools
import logging
import numpy as np
from ..sampling import subsample_labels
import wsummary
from object_detection2.standard_names import *
from object_detection2.datadef import *
import wnn

logger = logging.getLogger(__name__)

'''
this function get exactly candiate_nr boxes by the heristic method
firt remove boxes by nums, and then get the top candiate_nr boxes, if there not enough boxes after nms,
the boxes in the front will be add to result.
class_prediction:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率或与概率正相关的指标(如logits)
shape为[batch_size,X,num_classes]
bboxes_regs:模型预测的每个proposal_bboxes/anchro boxes/default boxes到目标真实bbox的回归参数
shape为[batch_size,X,4](classes_wise=Flase)或者(batch_size,X,num_classes,4](classes_wise=True)
proposal_bboxes:候选box
shape为[batch_size,X,4]
threshold:选择class_prediction的阀值
nms_threshold: nms阀值
candiate_nr:选择proposal_bboxes中预测最好的candiate_nr个bboxes进行筛选
limits:[4,2],用于对回归参数的范围进行限制，分别对应于cy,cx,h,w的回归参数，limits的值对应于prio_scaling=[1,1,1,1]是的设置
prio_scaling:在encode_boxes以及decode_boxes是使用的prio_scaling参数
anchors_num_per_level: anchors num per level
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
class_prediction:模型预测的每个proposal_bboxes/anchro boxes/default boxes所对应的类别的概率或与概率正相关的指标(如logits)
shape为[batch_size,X,num_classes]
bboxes_regs:模型预测的每个proposal_bboxes/anchro boxes/default boxes到目标真实bbox的回归参数
shape为[batch_size,X,4](classes_wise=Flase)或者(batch_size,X,num_classes,4](classes_wise=True)
proposal_bboxes:候选box
shape为[batch_size,X,4]
threshold:选择class_prediction的阀值
nms_threshold: nms阀值
candiate_nr:选择proposal_bboxes中预测最好的candiate_nr个bboxes进行筛选
limits:[4,2],用于对回归参数的范围进行限制，分别对应于cy,cx,h,w的回归参数，limits的值对应于prio_scaling=[1,1,1,1]是的设置
prio_scaling:在encode_boxes以及decode_boxes是使用的prio_scaling参数
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


def rpn_losses(
    gt_objectness_logits,
    gt_anchor_deltas,
    pred_objectness_logits,
    pred_anchor_deltas,
):
    localization_loss = tf.losses.huber_loss(
        pred_anchor_deltas,
        gt_anchor_deltas,
        reduction=tf.losses.Reduction.SUM,
        loss_collection=None
    )

    objectness_loss = tf.losses.sigmoid_cross_entropy(
        logits=tf.expand_dims(pred_objectness_logits,1),
        multi_class_labels=tf.cast(tf.expand_dims(gt_objectness_logits,axis=1),tf.float32),
        reduction=tf.losses.Reduction.SUM,
        loss_collection=None
    )
    return objectness_loss, localization_loss


class RPNOutputs(object):
    def __init__(
        self,
        box2box_transform,
        anchor_matcher,
        batch_size_per_image,
        positive_fraction,
        pred_objectness_logits,
        pred_anchor_deltas,
        anchors,
        gt_boxes=None,
        gt_length=None,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for
                anchor-proposal transformations.
            anchor_matcher (Matcher): :class:`Matcher` instance for matching anchors to
                ground-truth boxes; used to determine training labels.
            batch_size_per_image (int): number of proposals to sample when training
            positive_fraction (float): target fraction of sampled proposals that should be positive
            images (ImageList): :class:`ImageList` instance representing N input images
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for anchors.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, A*4, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
            anchors (list[list[Boxes]]): A list of N elements. Each element is a list of L
                Boxes. The Boxes at (n, l) stores the entire anchor array for feature map l in image
                n (i.e. the cell anchors repeated over all locations in feature map (n, l)).
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image
                boundary by more than boundary_thresh are not used in training. Set to a very large
                number or < 0 to disable this behavior. Only needed in training.
            gt_boxes (list[Boxes], optional): A list of N elements. Element i a Boxes storing
                the ground-truth ("gt") boxes for image i.
        """
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pred_objectness_logits = pred_objectness_logits
        self.pred_anchor_deltas = pred_anchor_deltas

        anchors = tf.concat(anchors,axis=0)
        anchors = tf.expand_dims(anchors,axis=0)
        self.anchors = anchors
        self.gt_boxes = gt_boxes
        self.gt_length = gt_length
        self.num_feature_maps = len(pred_objectness_logits)
        self.mid_results = {}

    def _get_ground_truth(self):
        """
        Returns:
            gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """
        res = self.anchor_matcher(self.anchors,self.gt_boxes,tf.ones(tf.shape(self.gt_boxes)[:2]),self.gt_length)
        gt_objectness_logits_i, scores, indices  = res
        self.mid_results['anchor_matcher'] = res

        gt_anchor_deltas = self.box2box_transform.get_deltas(self.anchors,self.gt_boxes,gt_objectness_logits_i,indices)
        #gt_objectness_logits_i为相应anchor box的标签
        return gt_objectness_logits_i, gt_anchor_deltas

    def losses(self):
        with tf.variable_scope("RPNLoss"):
            gt_objectness_logits, gt_anchor_deltas = self._get_ground_truth()
            #In a image, all anchors concated togather and sample, Detectron2 use the same strategy
            pos_idx, neg_idx = subsample_labels(gt_objectness_logits,
                                                self.batch_size_per_image, self.positive_fraction)

            batch_size = self.pred_objectness_logits[0].get_shape().as_list()[0]
            num_cell_anchors = self.pred_objectness_logits[0].get_shape().as_list()[-1] #RPN num classes==1
            box_dim = self.pred_anchor_deltas[0].get_shape().as_list()[-1]//num_cell_anchors
            pred_objectness_logits = [tf.reshape(x,[batch_size,-1]) for x in self.pred_objectness_logits]
            pred_objectness_logits = tf.concat(pred_objectness_logits,axis=1)
            pred_anchor_deltas = [tf.reshape(x,[batch_size,-1,box_dim]) for x in self.pred_anchor_deltas]
            pred_anchor_deltas = tf.concat(pred_anchor_deltas,axis=1)
            pred_objectness_logits = tf.reshape(pred_objectness_logits,[-1])
            pred_anchor_deltas = tf.reshape(pred_anchor_deltas,[-1,box_dim])

            if global_cfg.GLOBAL.DEBUG:
                with tf.device(":/cpu:0"):
                    with tf.name_scope("rpn_sampled_box"):
                        log_anchors = self.anchors*tf.ones([batch_size,1,1])
                        logmask = tf.reshape(pos_idx,[batch_size,-1])
                        wsummary.detection_image_summary_by_logmask(images=self.inputs[IMAGE],boxes=log_anchors,
                                                                    logmask=logmask)

            valid_mask = tf.logical_or(pos_idx,neg_idx)
            gt_objectness_logits = tf.reshape(gt_objectness_logits,[-1])
            gt_objectness_logits = tf.boolean_mask(gt_objectness_logits,valid_mask)
            pred_objectness_logits = tf.boolean_mask(pred_objectness_logits,valid_mask)
            gt_anchor_deltas = tf.reshape(gt_anchor_deltas,[-1,box_dim])
            gt_anchor_deltas = tf.boolean_mask(gt_anchor_deltas,pos_idx)
            pred_anchor_deltas = tf.boolean_mask(pred_anchor_deltas,pos_idx)
            objectness_loss, localization_loss = rpn_losses(
                gt_objectness_logits,
                gt_anchor_deltas,
                pred_objectness_logits,
                pred_anchor_deltas,
            )
            if global_cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.INFO:
                with tf.name_scope("RPNCorrectRatio"):
                    ratio = wnn.sigmoid_accuracy_ratio(logits=pred_objectness_logits,labels=gt_objectness_logits)
                tf.summary.scalar("rpn_accuracy_ratio",ratio)
            normalizer = 1.0 / (batch_size* self.batch_size_per_image)
            loss_cls = objectness_loss * normalizer  # cls: classification loss
            loss_loc = localization_loss * normalizer  # loc: localization loss
            losses = {"loss_rpn_cls": loss_cls, "loss_rpn_loc": loss_loc}
            wsummary.histogram_or_scalar(loss_cls,"rpn/cls_loss")
            wsummary.histogram_or_scalar(loss_loc,"rpn/loc_loss")

            return losses

    def predict_proposals(self):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B), where B is box dimension (4 or 5).
        """
        with tf.name_scope("predict_proposals"):
            batch_size = self.pred_objectness_logits[0].get_shape().as_list()[0]
            num_cell_anchors = self.pred_objectness_logits[0].get_shape().as_list()[-1]
            box_dim = self.pred_anchor_deltas[0].get_shape().as_list()[-1]//num_cell_anchors
            pred_anchor_deltas = [tf.reshape(x,[batch_size,-1,box_dim]) for x in self.pred_anchor_deltas]
            pred_anchor_deltas = tf.concat(pred_anchor_deltas,axis=1)
            proposals = self.box2box_transform.apply_deltas(deltas=pred_anchor_deltas,boxes=self.anchors)
            return proposals

    def predict_objectness_logits(self):
        """
        Return objectness logits in the same format as the proposals returned by
        :meth:`predict_proposals`.

        Returns:
            pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A).
        """
        with tf.name_scope("predict_objectness_logits"):
            batch_size = self.pred_objectness_logits[0].get_shape().as_list()[0]
            pred_objectness_logits = [tf.reshape(x,[batch_size,-1]) for x in self.pred_objectness_logits]
            pred_objectness_logits = tf.concat(pred_objectness_logits,axis=1)
            return pred_objectness_logits
