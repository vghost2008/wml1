# coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
import basic_tftools as btf
import wnn
from basic_tftools import channel
import functools
import wtfop.wtfop_ops as wop
import object_detection2.bboxes as odbox
from object_detection2.standard_names import *
import wmodule
from object_detection2.modeling.onestage_heads.onestage_tools import *
from object_detection2.datadef import *
from object_detection2.config.config import global_cfg
from object_detection2.modeling.build import HEAD_OUTPUTS
import object_detection2.wlayers as odl
from object_detection2.modeling.box_regression import CenterNet2Box2BoxTransform
import numpy as np
from object_detection2.data.dataloader import DataLoader
import wsummary
from functools import partial
import wnn

slim = tf.contrib.slim

@HEAD_OUTPUTS.register()
class FairMOTOutputs(wmodule.WChildModule):
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
            gt_boxes: [B,N,4] (ymin,xmin,ymax,xmax)
            gt_labels: [B,N]
            gt_length: [B]
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
        self.box2box_transform2 = CenterNet2Box2BoxTransform(num_classes=global_cfg.MODEL.MOT.FAIR_MOT_NUM_CLASSES,
                                                             k=self.k)


    def _get_ground_truth(self):
        """
        Returns:
        """
        res = []

        for i,outputs in enumerate(self.head_outputs):
            shape = wmlt.combined_static_and_dynamic_shape(outputs['heatmaps_ct'])[1:3]
            t_res = self.box2box_transform.get_deltas(self.gt_boxes,
                                                tf.ones_like(self.gt_labels),
                                                self.gt_length,
                                                output_size=shape)
            g_heatmaps_c, _, _= wop.center2_boxes_encode(self.gt_boxes,
                                                         self.gt_labels,
                                                         self.gt_length,
                                                         shape,
                                                         global_cfg.MODEL.MOT.FAIR_MOT_NUM_CLASSES,
                                                         0.7)
            B,H,W,_ = btf.combined_static_and_dynamic_shape(g_heatmaps_c)
            bg_prob = tf.ones([B,H,W,1])*0.5
            heatmaps = tf.concat([bg_prob,g_heatmaps_c],axis=-1)
            classes = tf.math.argmax(heatmaps,axis=-1)
            t_res['id_classes'] = classes
            res.append(t_res)
        return res

    @wmlt.add_name_scope
    def losses(self):
        """
        Args:

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only.
        """
        all_encoded_datas = self._get_ground_truth()
        all_loss0 = []
        all_loss1 = []
        all_loss2 = []
        all_loss3 = []
        for i,(encoded_datas,head_outputs) in enumerate(zip(all_encoded_datas,self.head_outputs)):
            loss0 = wnn.focal_loss_for_heat_map(labels=encoded_datas["g_heatmaps_ct"],
                                    logits=head_outputs["heatmaps_ct"],scope="ct_loss",
                                    alpha=self.cfg.LOSS_ALPHA,
                                    beta=self.cfg.LOSS_BETA,
                                    pos_threshold=self.cfg.LOSS_POS_THRESHOLD)
            tmp_w = tf.reduce_sum(encoded_datas['g_offset_mask'])+1e-3
            offset_loss = tf.reduce_sum(tf.abs((encoded_datas['g_offset']-head_outputs['offset'])*encoded_datas['g_offset_mask']))/tmp_w
            tmp_w = tf.reduce_sum(encoded_datas['g_hw_mask'])+1e-3
            hw_loss = tf.reduce_sum(tf.abs((encoded_datas['g_hw']-head_outputs['hw'])*encoded_datas['g_hw_mask']))/tmp_w
            loss3_mask = tf.greater(encoded_datas['id_classes'],0)
            loss3_mask = tf.reshape(loss3_mask,[-1])
            id_data = head_outputs['id_embedding']
            id_data = tf.reshape(id_data,[-1,btf.channel(id_data)])
            id_data = tf.boolean_mask(id_data,loss3_mask)
            id_data = slim.fully_connected(id_data,global_cfg.MODEL.MOT.FAIR_MOT_NUM_CLASSES+1,
                                           activation_fn=None,
                                           normalizer_fn=None,scope="trans_id_embedding")
            id_gt_lasses = encoded_datas['id_classes']
            id_gt_lasses = tf.reshape(id_gt_lasses,[-1])
            id_gt_lasses = tf.boolean_mask(id_gt_lasses,loss3_mask)
            #id_data = tf.Print(id_data,["X:",tf.reduce_max(id_gt_lasses),global_cfg.MODEL.MOT.FAIR_MOT_NUM_CLASSES,tf.reduce_sum(tf.cast(loss3_mask,tf.float32))],summarize=1000)
            loss3 = tf.losses.sparse_softmax_cross_entropy(labels=id_gt_lasses,
                                                           logits=id_data,
                                                           reduction=tf.losses.Reduction.MEAN,
                                                           loss_collection=None)
            all_loss0.append(loss0)
            all_loss1.append(offset_loss)
            all_loss2.append(hw_loss)
            all_loss3.append(loss3)

        loss0 = tf.add_n(all_loss0)
        loss1 = tf.add_n(all_loss1)*self.cfg.LOSS_LAMBDA_OFFSET
        loss2 = tf.add_n(all_loss2)*self.cfg.LOSS_LAMBDA_SIZE
        loss3 = tf.add_n(all_loss3)*global_cfg.MODEL.MOT.FAIR_MOT_LOSS_LAMBDA_ID_EMBEDDING

        loss_dict = {"heatmaps_ct_loss": loss0,
                "offset_loss": loss1,
                "hw_loss":loss2,
                "id_embedding_loss":loss3,
                }
        for k,v in loss_dict.items():
            tf.summary.scalar(k,v)
        det_loss = loss0+loss1+loss2
        id_loss = loss3
        w1 = tf.get_variable("w1",shape=(),
                             initializer=tf.constant_initializer(-1.85),
                             dtype=tf.float32,trainable=True)
        w2 = tf.get_variable("w2",shape=(),
                             initializer=tf.constant_initializer(-1.05),
                             dtype=tf.float32,trainable=True)
        #w1 = tf.Print(w1,["w1",w1,"w2",w2,det_loss,id_loss],summarize=1000)
        loss = det_loss*tf.exp(-w1)+id_loss*tf.exp(-w2)+(w1+w2)
        tf.summary.scalar("det_loss_weight",tf.exp(-w1))
        tf.summary.scalar("id_loss_weight",tf.exp(-w2))
        tf.summary.scalar("det_loss",tf.exp(-w1)*det_loss)
        tf.summary.scalar("id_loss",tf.exp(-w2)*id_loss)
        return {'loss':loss}


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
        assert len(head_outputs)==1,f"Error head outputs len {len(head_outputs)}"
        nms = partial(odl.boxes_nms,threshold=self.nms_threshold)
        bboxes,clses, scores,ids,length = self.get_box_in_a_single_layer(head_outputs[0],self.cfg.SCORE_THRESH_TEST)
        bboxes, labels, nms_indexs, lens = odl.batch_nms_wrapper(bboxes, clses, length, confidence=None,
                                  nms=nms,
                                  k=self.max_detections_per_image,
                                  sort=True)
        scores = wmlt.batch_gather(scores,nms_indexs)
        ids = wmlt.batch_gather(ids,nms_indexs)

        outdata = {RD_BOXES:bboxes,RD_LABELS:labels,RD_PROBABILITY:scores,RD_LENGTH:lens,
                   RD_ID:ids}
        if global_cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
            wsummary.detection_image_summary(images=inputs[IMAGE],
                                             boxes=outdata[RD_BOXES],
                                             classes=outdata[RD_LABELS],
                                             lengths=outdata[RD_LENGTH],
                                             scores=outdata[RD_PROBABILITY],
                                             name="FairMOTDetOutput",
                                             category_index=DataLoader.category_index)
        return outdata

    @wmlt.add_name_scope
    def get_box_in_a_single_layer(self,datas,threshold):
        bboxes,clses,scores,indexs = self.box2box_transform.apply_deltas(datas)
        ids = datas['id_embedding']
        B,H,W,C = btf.combined_static_and_dynamic_shape(ids)
        ids = tf.reshape(ids,[B,H*W,C])
        ids = wmlt.batch_gather(ids,indexs)
        mask = tf.cast(tf.greater_equal(scores,threshold),tf.int32)
        length = tf.reduce_sum(mask,axis=-1)
        return bboxes,clses,scores,ids,length
