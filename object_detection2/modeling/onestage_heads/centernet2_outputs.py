# coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
import wnn
from basic_tftools import channel
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
from functools import partial


@HEAD_OUTPUTS.register()
class CenterNet2Outputs(wmodule.WChildModule):
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
        self.score_threshold = cfg.SCORE_THRESH_TEST
        self.nms_threshold = cfg.NMS_THRESH_TEST
        self.max_detections_per_image = max_detections_per_image
        self.box2box_transform = box2box_transform
        self.head_outputs = head_outputs
        self.k = self.cfg.K
        self.size_threshold = self.cfg.SIZE_THRESHOLD
        self.dis_threshold = self.cfg.DIS_THRESHOLD
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
        res = []
        for i,outputs in enumerate(self.head_outputs):
            shape = wmlt.combined_static_and_dynamic_shape(outputs['heatmaps_ct'])[1:3]
            t_res = self.box2box_transform.get_deltas(self.gt_boxes,
                                                self.gt_labels,
                                                self.gt_length,
                                                output_size=shape)
            res.append(t_res)
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
        all_encoded_datas = self._get_ground_truth()
        all_loss0 = []
        all_loss1 = []
        all_loss2 = []
        for i,(encoded_datas,head_outputs) in enumerate(zip(all_encoded_datas,self.head_outputs)):
            loss0 = self.focal_loss(labels=encoded_datas["g_heatmaps_ct"],
                                    logits=head_outputs["heatmaps_ct"],scope="ct_loss",
                                    alpha=self.cfg.LOSS_ALPHA,
                                    beta=self.cfg.LOSS_BETA,
                                    pos_threshold=self.cfg.LOSS_POS_THRESHOLD)
            tmp_w = tf.reduce_sum(encoded_datas['g_offset_mask'])+1e-3
            offset_loss = tf.reduce_sum(tf.abs((encoded_datas['g_offset']-head_outputs['offset'])*encoded_datas['g_offset_mask']))/tmp_w
            tmp_w = tf.reduce_sum(encoded_datas['g_hw_mask'])+1e-3
            hw_loss = tf.reduce_sum(tf.abs((encoded_datas['g_hw']-head_outputs['hw'])*encoded_datas['g_hw_mask']))/tmp_w
            all_loss0.append(loss0)
            all_loss1.append(offset_loss)
            all_loss2.append(hw_loss)

        loss0 = tf.add_n(all_loss0)
        loss1 = tf.add_n(all_loss1)*self.cfg.LOSS_LAMBDA_OFFSET
        loss2 = tf.add_n(all_loss2)*self.cfg.LOSS_LAMBDA_SIZE

        return {"heatmaps_ct_loss": loss0,
                "offset_loss": loss1,
                "hw_loss":loss2}

    @staticmethod
    def focal_loss(labels,logits,pos_threshold=0.99,alpha=2,beta=4,scope=None):
        with tf.name_scope(scope,"focal_loss"):
            zeros = tf.zeros_like(labels)
            ones = tf.ones_like(labels)
            num_pos = tf.reduce_sum(tf.where(tf.greater_equal(labels, pos_threshold), ones, zeros))

            # loss=tf.reduce_mean(tf.log(logits))
            probs = tf.nn.sigmoid(logits)
            pos_weight = tf.where(tf.greater_equal(labels, pos_threshold), ones - probs, zeros)
            neg_weight = tf.where(tf.less(labels, pos_threshold), probs, zeros)
            '''
            用于保证数值稳定性，log(sigmoid(x)) = log(1/(1+e^-x) = -log(1+e^-x) = x-x-log(1+e^-x) = x-log(e^x +1)
            pos_loss = tf.where(tf.less(logits,0),logits-tf.log(tf.exp(logits)+1),tf.log(probs))
            '''
            pure_pos_loss = tf.minimum(logits,0)-tf.log(1+tf.exp(-tf.abs(logits)))
            pos_loss = pure_pos_loss*tf.pow(pos_weight, alpha)
            pos_loss = tf.reduce_sum(pos_loss)
            '''
            用于保证数值稳定性
            '''
            pure_neg_loss = -tf.nn.relu(logits)-tf.log(1+tf.exp(-tf.abs(logits)))
            neg_loss = tf.pow((1 - labels), beta) * tf.pow(neg_weight, alpha) * pure_neg_loss
            neg_loss = tf.reduce_sum(neg_loss)
            loss = -(pos_loss + neg_loss) / (num_pos + tf.convert_to_tensor(1e-4))
            tf.summary.scalar("neg_loss",neg_loss)
            tf.summary.scalar("pos_loss",pos_loss)
            #loss = tf.Print(loss,["mloss",probs,pos_weight,neg_weight,pos_loss,neg_loss,num_pos],summarize=100)
            return loss

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
        self.inputs = inputs
        all_bboxes = []
        all_scores = []
        all_clses = []
        all_length = []
        img_size = tf.shape(inputs[IMAGE])[1:3]
        assert len(head_outputs)==1,f"Error head outputs len {len(head_outputs)}"
        nms = partial(odl.boxes_nms,threshold=self.nms_threshold)
        bboxes,clses, scores,length = self.get_box_in_a_single_layer(head_outputs[0],self.cfg.SCORE_THRESH_TEST)
        bboxes, labels, nms_indexs, lens = odl.batch_nms_wrapper(bboxes, clses, length, confidence=None,
                                  nms=nms,
                                  k=self.max_detections_per_image,
                                  sort=True)
        scores = wmlt.batch_gather(scores,nms_indexs)

        outdata = {RD_BOXES:bboxes,RD_LABELS:labels,RD_PROBABILITY:scores,RD_LENGTH:lens}
        if global_cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
            wsummary.detection_image_summary(images=inputs[IMAGE],
                                             boxes=outdata[RD_BOXES],
                                             classes=outdata[RD_LABELS],
                                             lengths=outdata[RD_LENGTH],
                                             scores=outdata[RD_PROBABILITY],
                                             name="CenterNetOutput",
                                             category_index=DataLoader.category_index)
        return outdata

    @wmlt.add_name_scope
    def get_box_in_a_single_layer(self,datas,threshold):
        bboxes,clses,scores,_ = self.box2box_transform.apply_deltas(datas)
        mask = tf.cast(tf.greater_equal(scores,threshold),tf.int32)
        length = tf.reduce_sum(mask,axis=-1)
        return bboxes,clses,scores,length