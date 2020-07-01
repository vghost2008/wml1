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
            shape = wmlt.combined_static_and_dynamic_shape(outputs['heatmaps_tl'])[1:3]
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
        all_offset_loss = []
        all_embeading_loss = []
        for i,outputs in enumerate(self.head_outputs):
            encoded_datas = all_encoded_datas[i]
            head_outputs = self.head_outputs[i]
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
            all_loss0.append(loss0)
            all_loss1.append(loss1)
            all_loss2.append(loss2)
            all_offset_loss.append(offset_loss)
            all_embeading_loss.append(embeading_loss)

        loss0 = tf.add_n(all_loss0)
        loss1 = tf.add_n(all_loss1)
        loss2 = tf.add_n(all_loss2)
        offset_loss = tf.add_n(all_offset_loss)
        embeading_loss= tf.add_n(all_embeading_loss)

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

        all_bboxes = []
        all_scores = []
        all_clses = []
        all_length = []
        img_size = tf.shape(inputs[IMAGE])[1:3]
        for datas in head_outputs:
            bboxes, scores, clses, length = self.get_box_in_a_single_layer(datas,self.topk_candidates,img_size)
            all_bboxes.append(bboxes)
            all_scores.append(scores)
            all_clses.append(clses)
            all_length.append(length)

        with tf.name_scope(f"merge_all_boxes"):
            bboxes,_ = wmlt.batch_concat_with_length(all_bboxes,all_length)
            scores,_ = wmlt.batch_concat_with_length(all_scores,all_length)
            clses,length = wmlt.batch_concat_with_length(all_clses,all_length)

            nms = functools.partial(wop.boxes_nms, threshold=self.nms_threshold,
                                    classes_wise=True,
                                    k=self.max_detections_per_image)
            bboxes, labels, nms_indexs, lens = odl.batch_nms_wrapper(bboxes, clses, length, confidence=None,
                                  nms=nms,
                                  k=self.max_detections_per_image,
                                  sort=True)
            scores = wmlt.batch_gather(scores,nms_indexs)

        return {RD_BOXES:bboxes,RD_LABELS:labels,RD_PROBABILITY:scores,RD_LENGTH:lens}

    @staticmethod
    def pixel_nms(heat,kernel=[3,3],threshold=0.3):
        hmax=tf.nn.max_pool(heat,ksize=[1]+kernel+[1],strides=[1,1,1,1],padding='SAME')
        mask=tf.cast(tf.logical_and(tf.equal(hmax,heat),tf.greater(hmax,threshold)),tf.float32)
        return mask*heat

    @staticmethod
    @wmlt.add_name_scope
    def _topk(scores,K=100):
        B,H,W,C = wmlt.combined_static_and_dynamic_shape(scores)
        scores = tf.reshape(scores,[B,-1])
        topk_scores,topk_inds = tf.nn.top_k(scores,k=K)
        topk_classes = topk_inds%C
        topk_inds = topk_inds//C
        topk_ys = tf.cast(topk_inds//W,tf.float32)
        topk_xs = tf.cast(topk_inds%W,tf.float32)
        return topk_scores,topk_inds,topk_classes,topk_ys,topk_xs


    @wmlt.add_name_scope
    def get_box_in_a_single_layer(self,datas,num_dets,img_size):
        '''
        '''
        h_tl = tf.nn.sigmoid(datas['heatmaps_tl'])
        h_br  = tf.nn.sigmoid(datas['heatmaps_br'])
        h_ct = tf.nn.sigmoid(datas['heatmaps_ct'])

        B,H,W,C = wmlt.combined_static_and_dynamic_shape(h_tl)

        h_tl = self.pixel_nms(h_tl, threshold=self.nms_threshold)
        h_br = self.pixel_nms(h_br, threshold=self.nms_threshold)
        h_ct = self.pixel_nms(h_ct, threshold=self.nms_threshold)
        tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = self._topk(h_tl, K=self.k)
        br_scores, br_inds, br_clses, br_ys, br_xs = self._topk(h_br, K=self.k)
        ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = self._topk(h_ct, K=self.k)
        K = self.k
        tl_ys = tf.tile(tf.reshape(tl_ys,[B,K,1]),[1,1,K])
        tl_xs = tf.tile(tf.reshape(tl_xs,[B,K,1]),[1,1,K])
        br_ys = tf.tile(tf.reshape(br_ys,[B,1,K]),[1,K,1])
        br_xs = tf.tile(tf.reshape(br_xs,[B,1,K]),[1,K,1])
        ct_ys = tf.reshape(ct_ys,[B,K])
        ct_xs = tf.reshape(ct_xs,[B,K])
        if 'offset_tl' in datas:
            tl_regr = wmlt.batch_gather(datas['offset_tl'],tl_inds)
            br_regr = wmlt.batch_gather(datas['offset_br'],br_inds)
            ct_regr = wmlt.batch_gather(datas['offset_ct'],br_inds)
            tl_regr = tf.reshape(tl_regr,[B,K,1,2])
            br_regr = tf.reshape(br_regr,[B,1,K,2])
            ct_regr = tf.reshape(ct_regr,[B,K,2])
            tl_xs = tl_xs + tl_regr[...,0]
            tl_ys = tl_ys + tl_regr[...,1]
            br_xs = br_xs + br_regr[...,0]
            br_ys = br_ys + br_regr[...,1]
            ct_xs = ct_xs + ct_regr[...,0]
            ct_ys = ct_ys + ct_regr[...,1]

        bboxes = tf.stack([tl_ys,tl_xs,br_ys,br_xs],axis=-1)
        tl_tag = wmlt.batch_gather(datas['tag_tl'],tl_inds)
        br_tag = wmlt.batch_gather(datas['tag_br'],br_inds)
        tl_tag = tf.expand_dims(tl_tag,axis=2)
        br_tag = tf.expand_dims(br_tag,axis=1)
        tl_tag = tf.tile(tl_tag,[1,1,K,1])
        br_tag = tf.tile(br_tag,[1,K,1,1])
        dists = tf.abs(tl_tag-br_tag)
        dists = tf.squeeze(dists,axis=-1)
        dis_inds = (dists>self.dis_threshold)

        tl_scores = tf.tile(tf.reshape(tl_scores,[B,K,1]),[1,1,K])
        br_scores = tf.tile(tf.reshape(br_scores,[B,1,K]),[1,K,1])
        scores = (tl_scores+br_scores)/2

        tl_clses = tf.tile(tf.reshape(tl_clses,[B,K,1]),[1,1,K])
        br_clses = tf.tile(tf.reshape(br_clses,[B,1,K]),[1,K,1])
        cls_inds = tf.not_equal(tl_clses,br_clses)

        width_inds = (br_xs<tl_xs)
        height_inds = (br_ys<tl_ys)

        all_inds = tf.logical_or(cls_inds,dis_inds)
        all_inds = tf.logical_or(all_inds,width_inds)
        all_inds = tf.logical_or(all_inds,height_inds)
        scores = tf.where(all_inds,tf.zeros_like(scores),scores)
        scores,inds = tf.nn.top_k(tf.reshape(scores,[B,-1]),num_dets)

        bboxes = tf.reshape(bboxes,[B,-1,4])
        bboxes = wmlt.batch_gather(bboxes,inds)

        clses = tf.reshape(tl_clses,[B,-1])
        clses = wmlt.batch_gather(clses,inds)

        '''tl_scores = tf.reshape(tl_scores,[B,-1,1])
        tl_scores = wmlt.batch_gather(tl_scores,inds)

        br_scores = tf.reshape(br_scores,[B,-1,1])
        br_scores = wmlt.batch_gather(br_scores,inds)'''

        ct = tf.stack([ct_ys, ct_xs], axis=-1)
        bboxes = odbox.tfabsolutely_boxes_to_relative_boxes(bboxes,width=W,height=H)
        sizes = tf.convert_to_tensor(self.size_threshold,dtype=tf.float32)
        relative_size = sizes*tf.rsqrt(tf.cast(img_size[0]*img_size[1],tf.float32))
        _,box_nr,_ = wmlt.combined_static_and_dynamic_shape(bboxes)
        length = tf.ones([B],tf.int32)*box_nr
        center_mask = wop.center_boxes_filter(bboxes=bboxes,center_points=ct,
                                        size_threshold=relative_size,
                                        bboxes_length=length,
                                        nrs=[3,5])
        def fn(bboxes,scores,clses,mask):
            mask = tf.logical_and(mask,tf.greater(scores,self.score_threshold))
            bboxes = tf.boolean_mask(bboxes,mask)
            scores = tf.boolean_mask(scores,mask)
            clses = tf.boolean_mask(clses,mask)
            len = tf.reduce_sum(tf.cast(mask,tf.int32))
            bboxes = tf.pad(bboxes,[[0,box_nr-len],[0,0]])
            scores = tf.pad(scores,[[0,box_nr-len]])
            clses = tf.pad(clses,[[0,box_nr-len]])
            return bboxes,scores,clses,len

        bboxes,scores,clses,length  = tf.map_fn(lambda x:fn(x[0],x[1],x[2],x[3]),
                                                elems=(bboxes,scores,clses,center_mask),
                                                dtype=(tf.float32,tf.float32,tf.int32,tf.int32))

        return bboxes,scores,clses,length
