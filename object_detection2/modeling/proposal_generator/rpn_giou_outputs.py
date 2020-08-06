import tensorflow as tf
import wml_tfutils as wmlt
import wtfop.wtfop_ops as wop
from object_detection2.config.config import global_cfg
from object_detection2.modeling.build import HEAD_OUTPUTS
import itertools
import logging
import numpy as np
from ..sampling import subsample_labels
import wsummary
from object_detection2.standard_names import *
from object_detection2.datadef import *
import object_detection2.wlayers as odl
import wnn
from .rpn_toolkit import *

logger = logging.getLogger(__name__)

def rpn_losses_giou(
    gt_objectness_logits,
    gt_anchor_deltas,
    pred_objectness_logits,
    pred_anchor_deltas,
):
    reg_loss_sum = 1.0 - odl.giou(pred_anchor_deltas, gt_anchor_deltas)
    localization_loss = tf.reduce_sum(reg_loss_sum)

    objectness_loss = tf.losses.sigmoid_cross_entropy(
        logits=tf.expand_dims(pred_objectness_logits,1),
        multi_class_labels=tf.cast(tf.expand_dims(gt_objectness_logits,axis=1),tf.float32),
        reduction=tf.losses.Reduction.SUM,
        loss_collection=None
    )
    return objectness_loss, localization_loss


@HEAD_OUTPUTS.register()
class RPNGIOUOutputs(object):
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
            anchors (list[Tensor]): A list of Tensor. Each element is a Tensor with shape [?,4]
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
        self.anchors_lens = [tf.shape(x)[0] for x in anchors]

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
        res = self.anchor_matcher(self.anchors,self.gt_boxes,tf.ones(tf.shape(self.gt_boxes)[:2]),self.gt_length,
                                  boxes_len=self.anchors_lens)
        gt_objectness_logits_i, scores, indices  = res
        self.mid_results['anchor_matcher'] = res

        gt_anchor_deltas = wmlt.batch_gather(self.gt_boxes,tf.maximum(indices,0))
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
            pred_anchor_deltas = tf.concat(pred_anchor_deltas,axis=1) #shape=[B,-1,4]
            pred_objectness_logits = tf.reshape(pred_objectness_logits,[-1])
            anchors = tf.tile(self.anchors,[batch_size,1,1])
            anchors = tf.reshape(anchors,[-1,box_dim])
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
            anchors = tf.boolean_mask(anchors,pos_idx)
            
            pred_anchor_deltas = self.box2box_transform.apply_deltas(deltas=pred_anchor_deltas,boxes=anchors)
            objectness_loss, localization_loss = rpn_losses_giou(
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
