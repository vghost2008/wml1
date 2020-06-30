# coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
import wnn
import functools
import wtfop.wtfop_ops as wop
import object_detection2.bboxes as odbox
from object_detection2.standard_names import *
import wmodule
from .onestage_tools import *
from object_detection2.datadef import *
from object_detection2.config.config import global_cfg
from object_detection2.modeling.build import HEAD_OUTPUTS
import object_detection2.wlayers as odl
import numpy as np
from object_detection2.data.dataloader import DataLoader
import wsummary


@HEAD_OUTPUTS.register()
class CenterNetOutputs(wmodule.WChildModule):
    def __init__(
            self,
            cfg,
            parent,
            box2box_transform,
            head_outputs,
            gt_boxes=None,
            gt_labels=None,
            gt_length=None,
            max_detections_per_image=100,
            **kwargs,
    ):
        """
        Args:
            cfg: Only the child part
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for
                anchor-proposal transformations.
            images (ImageList): :class:`ImageList` instance representing N input images
            gt_boxes (list[Boxes], optional): A list of N elements. Element i a Boxes storing
                the ground-truth ("gt") boxes for image i.
        """
        super().__init__(cfg, parent=parent, **kwargs)
        self.num_classes = cfg.NUM_CLASSES
        self.topk_candidates = cfg.TOPK_CANDIDATES_TEST
        self.score_threshold = cfg.SCORE_THRESH_TEST
        self.nms_threshold = cfg.NMS_THRESH_TEST
        self.max_detections_per_image = max_detections_per_image
        self.box2box_transform = box2box_transform
        self.head_outputs = head_outputs

        self.gt_boxes = gt_boxes
        self.gt_labels = gt_labels
        self.gt_length = gt_length
        self.mid_results = {}

    def _get_ground_truth(self):
        """
        Returns:
            gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """
        shape = wmlt.combined_static_and_dynamic_shape(self.head_outputs['heatmaps_tl'])[1:3]
        res = self.box2box_transform.get_deltas(self.gt_boxes,
                                                self.gt_labels,
                                                self.gt_length,
                                                output_size=shape)
        return res

    @wmlt.add_name_scope
    def losses(self):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`CenterNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`CenterNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        encoded_datas = self._get_ground_truth()
        head_outputs = self.head_outputs
        loss0 = tf.reduce_mean(self.focal_loss(labels=encoded_datas["g_heatmaps_tl"],
                                               logits=head_outputs["heatmaps_tl"]))
        loss1 = tf.reduce_mean(self.focal_loss(labels=encoded_datas["g_heatmaps_br"],
                                               logits=head_outputs["heatmaps_br"]))
        loss2 = tf.reduce_mean(self.focal_loss(labels=encoded_datas["g_heatmaps_ct"],
                                               logits=head_outputs["heatmaps_ct"]))
        offset0 = wmlt.batch_gather(head_outputs['offset_tl'],encoded_datas['g_index'][:,:,0])
        offset1 = wmlt.batch_gather(head_outputs['offset_br'],encoded_datas['g_index'][:,:,1])
        offset2 = wmlt.batch_gather(head_outputs['offset_ct'],encoded_datas['g_index'][:,:,2])
        offset = tf.concat([offset0,offset1,offset2],axis=2)
        offset_loss = tf.losses.huber_loss(labels=encoded_datas['g_offset'],
                                           predictions=offset,
                                           loss_collection=None,
                                           weights=tf.cast(tf.expand_dims(encoded_datas['g_index_mask'],-1),tf.float32))
        embeading_loss = self.ae_loss(head_outputs['tag_tl'],head_outputs['tag_br'],
                                      encoded_datas['g_index'],
                                      encoded_datas['g_index_mask'])

        return {"heatmaps_tl_loss": loss0,
                "heatmaps_br_loss": loss1,
                "heatmaps_ct_loss":loss2,
                "offset_loss":offset_loss,
                'embeading_loss':embeading_loss}

    @staticmethod
    def focal_loss(labels,logits):
        zeros = tf.zeros_like(labels)
        ones = tf.ones_like(labels)
        num_pos = tf.reduce_sum(tf.where(tf.equal(labels, 1), ones, zeros))
        loss = 0
        # loss=tf.reduce_mean(tf.log(logits))
        logits = tf.nn.sigmoid(logits)
        pos_weight = tf.where(tf.equal(labels, 1), ones - logits, zeros)
        neg_weight = tf.where(tf.less(labels, 1), logits, zeros)
        pos_loss = tf.reduce_sum(tf.log(logits) * tf.pow(pos_weight, 2))
        neg_loss = tf.reduce_sum(tf.pow((1 - labels), 4) * tf.pow(neg_weight, 2) * tf.log((1 - logits)))
        loss = loss - (pos_loss + neg_loss) / (num_pos + tf.convert_to_tensor(1e-4))
        return loss

    @staticmethod
    @wmlt.add_name_scope
    def ae_loss(tag0,tag1,index,mask):
        '''

        :param tag0: [B,N,C],top left tag
        :param tag1: [B,N,C], bottom right tag
        :param index: [B,M]
        :parma mask: [B,M]
        :return:
        '''
        with tf.name_scope("pull_loss"):
            num = tf.reduce_sum(tf.cast(mask,tf.float32))+1e-4
            tag0 = wmlt.batch_gather(tag0,index[:,:,0])
            tag1 = wmlt.batch_gather(tag1,index[:,:,1])
            tag_mean = (tag0+tag1)/2
            tag0 = tf.pow(tag0-tag_mean,2)/num
            tag0 = tf.reduce_sum(tf.boolean_mask(tag0,mask))
            tag1 = tf.pow(tag1-tag_mean,2)/num
            tag1 = tf.reduce_sum(tf.boolean_mask(tag1,mask))
            pull = tag0+tag1

        with tf.name_scope("push_loss"):
            neg_index = wop.make_neg_pair_index(mask)
            push_mask = tf.greater(neg_index,-1)
            neg_index = tf.nn.relu(neg_index)
            num = tf.reduce_sum(tf.cast(push_mask,tf.float32))+1e-4
            tag0 = wmlt.batch_gather(tag_mean,neg_index[:,:,0])
            tag1 = wmlt.batch_gather(tag_mean,neg_index[:,:,1])
            push = tf.reduce_sum(tf.nn.relu(1-tf.abs(tag0-tag1)))/num

        return pull+push


    @wmlt.add_name_scope
    def inference(self,inputs,head_outputs):
        """
        Arguments:
            inputs: same as CenterNet.forward's batched_inputs
        Returns:
            results:
            RD_BOXES: [B,N,4]
            RD_LABELS: [B,N]
            RD_PROBABILITY:[ B,N]
            RD_LENGTH:[B]
        """
        del inputs
        nms = functools.partial(wop.boxes_nms, threshold=self.nms_threshold)
        bboxes, labels, probs, indexs, lens = self._getBoxes(head_outputs=head_outputs,
                                                             cap_k=self.topk_candidates,
                                                             nms=nms)

        return {RD_BOXES:bboxes,RD_LABELS:labels,RD_PROBABILITY:probs,RD_LENGTH:lens}

    def _getBoxes(self,head_outputs,cap_k=200,nms=None):
        bboxes, labels, probs, indexs, lens = self.box2box_transform.apply_deltas(head_outputs)
        labels += 1
        bboxes,labels,nms_indexs,lens = odl.batch_nms_wrapper(bboxes,labels,lens,confidence=probs,nms=nms,k=cap_k,sort=True)
        indexs = tf.cond(tf.greater(tf.shape(indexs)[1],0),lambda:wmlt.batch_gather(indexs,nms_indexs),lambda:indexs)
        probs = tf.cond(tf.greater(tf.shape(probs)[1],0),lambda:wmlt.batch_gather(probs,nms_indexs),lambda:probs)
        return bboxes,labels,probs,indexs,len
