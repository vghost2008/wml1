#coding=utf-8
from thirdparty.registry import Registry
import wmodule
from object_detection2.modeling.matcher import Matcher
from object_detection2.modeling.box_regression import Box2BoxTransform
import tensorflow as tf
from thirdparty.nets.resnet_v1 import *
from object_detection2.modeling.poolers import ROIPooler
from .fast_rcnn import *
from .mask_head import *
from .box_head import build_box_head
import object_detection2.od_toolkit as od
import object_detection2.bboxes as bboxes
from object_detection2.standard_names import *
from wtfop.wtfop_ops import wpad
from object_detection2.datadef import *
from .keypoint_head import build_keypoint_head

slim = tf.contrib.slim


ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""


def build_roi_heads(cfg, *args,**kwargs):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, *args,**kwargs)


def select_foreground_proposals(proposals:EncodedData):
    """
    data in proposals's shape is like [batch_size,box_nr,...]
    return:
    [batch_size,box_nr]
    """
    gt_labels = proposals.gt_object_logits
    fg_selection_mask = tf.greater(gt_labels,0)
    return fg_selection_mask




class ROIHeads(wmodule.WChildModule):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, *args,**kwargs):
        super().__init__(cfg,*args,**kwargs)

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES     #不包含背景
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            allow_low_quality_matches=False,
            cfg=cfg,
            parent=self,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

    '''
    随机在输入中选择指定数量的box
    要求必须结果包含neg_nr+pos_nr个结果，如果总数小于这么大优先使用背影替代，然后使用随机替代
    注:Fast-RCNN原文中随机取25% IOU>0.5的目标, 75%IOU<=0.5的目标, 一个批次总共取128个目标，也就是每张图取128/batch_size个目标
    注:Detectron2中每个图像都会取512个目标，远远大于Faster-RCNN原文的数量, 比例与原文一样(25%的正样本)
    注:不使用sampling.subsample_labels是因为subsample_labels返回的数量可能小于neg_nr+pos_nr
    labels:[X],0表示背景,-1表示忽略，>0表示正样本
    keep_indices:[X],tf.bool
    返回:
    r_labels:[nr]
    r_indices:[nr]
    '''
    @staticmethod
    def sample_proposals(labels, neg_nr, pos_nr):
        nr = neg_nr + pos_nr
        r_indices = tf.range(0, tf.shape(labels)[0])
        keep_indices = tf.greater_equal(labels,0)
        labels = tf.boolean_mask(labels, keep_indices)
        r_indices = tf.boolean_mask(r_indices, keep_indices)
        total_neg_nr = tf.reduce_sum(tf.cast(tf.equal(labels, 0), tf.int32))
        total_pos_nr = tf.shape(labels)[0] - total_neg_nr

        with tf.name_scope("SelectRCNBoxes"):
            def random_select(labels, indices, size):
                with tf.name_scope("random_select"):
                    data_nr = tf.shape(labels)[0]
                    indexs = tf.range(data_nr)
                    indexs = wpad(indexs, [0, size - data_nr])
                    indexs = tf.random_shuffle(indexs)
                    indexs = tf.random_crop(indexs, [size])
                    labels = tf.gather(labels, indexs)
                    indices = tf.gather(indices, indexs)
                    return labels, indices

            # positive and negitave number satisfy the requement
            def selectRCNBoxesM0():
                with tf.name_scope("M0"):
                    n_mask = tf.equal(labels, 0)
                    p_mask = tf.logical_not(n_mask)
                    n_labels = tf.boolean_mask(labels, n_mask)
                    n_indices = tf.boolean_mask(r_indices, n_mask)
                    p_labels = tf.boolean_mask(labels, p_mask)
                    p_indices = tf.boolean_mask(r_indices, p_mask)
                    p_labels, p_indices = random_select(p_labels, p_indices, pos_nr)
                    n_labels, n_indices = random_select(n_labels, n_indices, neg_nr)

                    return tf.concat([p_labels, n_labels], axis=0), tf.concat([p_indices, n_indices], axis=0)

            # process the default situation
            def selectRCNBoxesM1():
                with tf.name_scope("M1"):
                    return random_select(labels, r_indices, nr)

            # positive dosen't satisfy the requement
            def selectRCNBoxesM2():
                with tf.name_scope("M2"):
                    n_mask = tf.equal(labels, 0)
                    p_mask = tf.logical_not(n_mask)
                    n_labels = tf.boolean_mask(labels, n_mask)
                    n_indices = tf.boolean_mask(r_indices, n_mask)
                    p_labels = tf.boolean_mask(labels, p_mask)
                    p_indices = tf.boolean_mask(r_indices, p_mask)
                    with tf.name_scope("Select"):
                        n_labels, n_indices = \
                            random_select(n_labels, n_indices, nr - total_pos_nr)
                    return tf.concat([p_labels, n_labels], axis=0), tf.concat([p_indices, n_indices], axis=0)

            # positive and negative is empty
            def selectRCNBoxesM3():
                with tf.name_scope("M3"):
                    # boxes = tf.constant([[0.0,0.0,0.001,0.001]],dtype=tf.float32)*tf.ones([nr,4],dtype=tf.float32)
                    # boxes_regs = tf.zeros_like(boxes,dtype=tf.float32)
                    labels = tf.constant([0]) * tf.ones([nr], dtype=tf.int32)
                    # scores = tf.ones_like(labels,dtype=tf.float32)
                    return labels, tf.zeros_like(labels)

            r_labels, r_indices = tf.case({
                tf.logical_and(total_pos_nr >= pos_nr, total_neg_nr >= neg_nr): selectRCNBoxesM0,
                tf.logical_and(tf.logical_and(total_pos_nr < pos_nr, total_neg_nr >= neg_nr),
                               total_pos_nr > 0): selectRCNBoxesM2,
                tf.equal(tf.shape(labels)[0], 0): selectRCNBoxesM3
            },
                default=selectRCNBoxesM1,
                exclusive=True)
            r_labels.set_shape([nr])
            r_indices.set_shape([nr])
            return r_labels, r_indices


    '''
    At the begining of training, most of the output bboxes (proposal_boxes) of rpn is negative, so there is no positive
    bboxes for rcn, in order to process this situation, we add the distored gtboxes to proposal_boxes, so this function 
    should be called between getProposalBoxes and encodeRCNBoxes
    Note that: Detectron2 also use the same strategy, but they juse simple add the gtboxes to the proposal_boxes
    '''
    def add_ground_truth_to_proposals(self,proposals,gtboxes,boxes_lens,nr=16,limits=[0.1,0.1,0.1,0.1]):
        with tf.name_scope("add_ground_truth_boxes_to_proposal_boxes"):
            boxes,_ = od.batched_random_select_boxes(gtboxes,boxes_lens,nr)
            if limits is not None:
                boxes = bboxes.random_distored_boxes(boxes,limits=limits,size=1,keep_org=True)
            proposals = tf.concat([proposals,boxes],axis=1)
        return proposals

    def label_and_sample_proposals(self, inputs,proposals,do_sample=True):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        with tf.name_scope("label_and_sample_proposals"):
            gt_boxes = inputs[GT_BOXES]
            gt_labels = inputs[GT_LABELS]
            gt_length = inputs[GT_LENGTH]
            # Augment proposals with ground-truth boxes.
            # In the case of learned proposals (e.g., RPN), when training starts
            # the proposals will be low quality due to random initialization.
            # It's possible that none of these initial
            # proposals have high enough overlap with the gt objects to be used
            # as positive examples for the second stage components (box head,
            # cls head, mask head). Adding the gt boxes to the set of proposals
            # ensures that the second stage components will have some positive
            # examples from the start of training. For RPN, this augmentation improves
            # convergence and empirically improves box AP on COCO by about 0.5
            # points (under one tested configuration).
            if self.proposal_append_gt:
                proposals = self.add_ground_truth_to_proposals(proposals,gt_boxes,gt_length,limits=None)
            res = self.proposal_matcher(proposals, gt_boxes,gt_labels, gt_length)
            gt_logits_i, scores, indices = res
            #gt_logits_i = tf.Print(gt_logits_i,[scores],name="XXXXX_matcher_scores",summarize=10000)
            #gt_logits_i = tf.Print(gt_logits_i,[gt_logits_i],name="XXXXX_matcher_scoresgtlogitsi",summarize=10000)

            if do_sample:
                pos_nr = int(self.batch_size_per_image*self.positive_sample_fraction)
                neg_nr = self.batch_size_per_image-pos_nr
                batch_size = gt_logits_i.get_shape().as_list()[0]

                gt_logits_i, rcn_indices = \
                    tf.map_fn(lambda x: self.sample_proposals(x,neg_nr=neg_nr, pos_nr=pos_nr),
                              elems=(gt_logits_i),
                              dtype=(tf.int32, tf.int32),
                              back_prop=False,
                              parallel_iterations=batch_size)
                #gt_logits_i = tf.Print(gt_logits_i,[scores],name="XXXXX_matcher_scores",summarize=10000)
                #gt_logits_i = tf.Print(gt_logits_i,[gt_logits_i],name="XXXXX_matcher_scores",summarize=10000)

                indices = tf.stop_gradient(wmlt.batch_gather(indices, rcn_indices))
                proposals = tf.stop_gradient(wmlt.batch_gather(proposals, rcn_indices))
                scores = wmlt.batch_gather(scores, rcn_indices)
            if self.cfg.GLOBAL.DEBUG:
                with tf.name_scope("label_and_sample_proposals_summary"):
                    logmask = tf.greater(gt_logits_i,0)
                    wsummary.detection_image_summary_by_logmask(images=inputs[IMAGE],
                                                                boxes=proposals,
                                                                classes=gt_logits_i,
                                                                scores=scores,
                                                                logmask=logmask,
                                                                name="label_and_sample_proposals_summary")
                    pgt_boxes = wmlt.batch_gather(inputs[GT_BOXES],tf.nn.relu(indices)) #background's indices is -1
                    wsummary.detection_image_summary_by_logmask(images=inputs[IMAGE],
                                                                boxes=pgt_boxes,
                                                                classes=gt_logits_i,
                                                                scores=scores,
                                                                logmask=logmask,
                                                                name="label_and_sample_proposals_summary_by_gtboxes")


            res = EncodedData(gt_logits_i,scores,indices,proposals,gt_boxes,gt_labels)

        return res


    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            results (list[Instances]): length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            losses (dict[str->Tensor]):
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg,parent,**kwargs):
        super().__init__(cfg,parent,**kwargs)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        bin_size = cfg.MODEL.ROI_BOX_HEAD.bin_size
        self.mask_on      = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            cfg=cfg.MODEL.ROI_BOX_HEAD,
            parent=self,
            output_size=pooler_resolution,
            bin_size=bin_size,
            pooler_type=pooler_type,
        )

        self.box_predictor = FastRCNNOutputLayers(cfg,parent=self,
            num_classes=self.num_classes, cls_agnostic_bbox_reg=self.cls_agnostic_bbox_reg,**kwargs
        )

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                parent=self,
                **kwargs
            )

    def res5_block(self, net):
        batch_norm_decay = self.cfg.MODEL.RESNETS.batch_norm_decay #0.999
        nr = 3
        use_batch_norm = True
        blocks = [
            resnet_utils.Block('block4', bottleneck, [{
                'depth': 2048,
                'depth_bottleneck': 512,
                'stride': 1
            }] * nr)
        ]
        with slim.arg_scope(resnet_arg_scope(batch_norm_decay=batch_norm_decay,
                                             is_training=self.is_training,
                                             use_batch_norm=use_batch_norm)):
            with tf.variable_scope("resnet_v1_101"):
                net = resnet_utils.stack_blocks_dense(
                        net, blocks)
        return net


    def _shared_roi_transform(self, features, boxes):
        '''
        返回的batch_size与box nr 合并到了新的batch_size这一维
        '''
        x = self.pooler(features, boxes)
        x = self.res5_block(x)
        return x

    def forward(self, inputs, features, proposals:ProposalsData):
        """
        See :class:`ROIHeads.forward`.
        """
        proposal_boxes = proposals[PD_BOXES]
        if self.is_training:
            proposals = self.label_and_sample_proposals(inputs,proposal_boxes)
            proposal_boxes = proposals.boxes

        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = tf.reduce_mean(box_features,axis=[1, 2],keep_dims=False)  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(feature_pooled)
        del feature_pooled

        outputs = FastRCNNOutputs(
            cfg=self.cfg,
            parent=self,
            box2box_transform=self.box2box_transform,
            pred_class_logits=pred_class_logits,
            pred_proposal_deltas=pred_proposal_deltas,
            proposals=proposals,
        )

        if self.is_training:
            del features
            losses = outputs.losses()
            if self.cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
                pred_instances = outputs.inference(
                    self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img,
                    proposal_boxes=proposals.boxes
                )
            else:
                pred_instances = {}
            if self.mask_on:
                fg_selection_mask = select_foreground_proposals(proposals)
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                '''
                feature in box_features's shape is [batch_size*box_nr,H,W,C]
                '''
                fg_selection_mask = tf.reshape(fg_selection_mask,[-1])
                mask_features = tf.boolean_mask(box_features,fg_selection_mask)
                del box_features
                mask_logits = self.mask_head(mask_features)
                losses["loss_mask"] = mask_rcnn_loss(inputs,mask_logits, proposals,fg_selection_mask)
            #return {}, losses
            return pred_instances, losses
        else:
            pred_instances = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances:RCNNResultsData):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.is_training

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, instances.boxes)
            mask_logits = self.mask_head(x)
            mask_rcnn_inference(mask_logits, instances)
        return instances


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, parent,*args,**kwargs):
        super().__init__(cfg, *args,parent=parent,**kwargs)
        self._init_box_head(cfg,*args,**kwargs)
        self._init_mask_head(cfg,*args,**kwargs)
        self._init_keypoint_head(cfg,*args,**kwargs)
        self.rcnn_outboxes = None

    def _init_box_head(self, cfg,*args,**kwargs):

        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        bin_size = cfg.MODEL.ROI_BOX_HEAD.bin_size
        '''
        是否将RCNN预测的结果放mask/keypoint 在分支输入的proposal box中，Detectron2所有的默认设置及配置文件中这一项都是False
        '''
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same

        self.box_pooler = ROIPooler(cfg=cfg.MODEL.ROI_BOX_HEAD,parent=self,
            output_size=pooler_resolution,
            bin_size=bin_size,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg,parent=self,*args,**kwargs
        )
        self.box_predictor = FastRCNNOutputLayers(cfg,parent=self,
            num_classes=self.num_classes, cls_agnostic_bbox_reg=self.cls_agnostic_bbox_reg,**kwargs
        )

    def _init_mask_head(self, cfg,*args,**kwargs):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        bin_size = cfg.MODEL.ROI_MASK_HEAD.bin_size
        # fmt: on

        self.mask_pooler = ROIPooler(cfg=cfg.MODEL.ROI_MASK_HEAD,parent=self,
            output_size=pooler_resolution,
            bin_size=bin_size,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg,
            parent=self,
            **kwargs
        )

    def _init_keypoint_head(self, cfg,*args,**kwargs):
        # fmt: off
        self.keypoint_on                         = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution                        = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_type                              = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        bin_size = cfg.MODEL.ROI_KEYPOINT_HEAD.bin_size
        self.normalize_loss_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.keypoint_loss_weight                = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(cfg=cfg.MODEL.ROI_KEYPOINT_HEAD,parent=self,
                                     output_size=pooler_resolution,
                                     bin_size=bin_size,
                                     pooler_type=pooler_type,
                                     )
        self.keypoint_head = build_keypoint_head(
            cfg, parent=self,*args,**kwargs
        )

    def forward(self, inputs, features, proposals: ProposalsData):
        """
        See :class:`ROIHeads.forward`.
        """
        proposals_boxes = proposals[PD_BOXES]
        if self.is_training:
            proposals = self.label_and_sample_proposals(inputs,proposals_boxes)

        features_list = [features[f] for f in self.in_features]

        if self.is_training:
            pred_instances,losses = self._forward_box(features_list, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            if self.train_on_pred_boxes:
                #proposals里面的box已经是采样的结果,无需再进行采样操作
                proposals = self.label_and_sample_proposals(inputs,proposals.boxes,do_sample=False)
            losses.update(self._forward_mask(inputs,features_list, proposals))
            losses.update(self._forward_keypoint(inputs,features_list, proposals))
            return pred_instances, losses
        else:
            pred_instances,_ = self._forward_box(features_list, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            mk_pred_instances = self.forward_with_given_boxes(inputs,features, pred_instances)
            pred_instances.update(mk_pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, inputs,features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.is_training
        features = [features[f] for f in self.in_features]

        instances = self._forward_mask(inputs,features, instances)
        instances = self._forward_keypoint(inputs,features, instances)
        return instances

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        if self.is_training:
            proposal_boxes = proposals.boxes #when training proposals's EncodedData
        else:
            proposal_boxes = proposals[PD_BOXES] #when inference proposals's a dict which is the outputs of RPN
        box_features = self.box_pooler(features, proposal_boxes)
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features

        outputs = FastRCNNOutputs(
            cfg=self.cfg,parent=self,
            box2box_transform=self.box2box_transform,
            pred_class_logits=pred_class_logits,
            pred_proposal_deltas=pred_proposal_deltas,
            proposals=proposals,
        )
        if self.is_training:
            if self.train_on_pred_boxes:
                pred_boxes = outputs.predict_boxes_for_gt_classes()
                self.rcnn_outboxes = pred_boxes
            if self.cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
                pred_instances = outputs.inference(
                    self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img,
                    proposal_boxes=proposals.boxes
                )
            else:
                pred_instances = {}
            return pred_instances,outputs.losses()
        else:
            pred_instances = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances,{}

    def _forward_mask(self, inputs,features, instances):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.is_training else instances

        if self.is_training:
            #when training, instance is EncodedData
            # The loss is only defined on positive proposals.
            fg_selection_mask = select_foreground_proposals(instances)
            proposal_boxes = instances.boxes
            mask_features = self.mask_pooler(features, proposal_boxes)
            fg_selection_mask = tf.reshape(fg_selection_mask,[-1])
            mask_features = tf.boolean_mask(mask_features, fg_selection_mask)
            mask_logits = self.mask_head(mask_features)
            return {"loss_mask": mask_rcnn_loss(inputs,mask_logits, instances,fg_selection_mask)}
        else:
            #when inference instances is RCNNResultsData
            pred_boxes = instances[RD_BOXES]
            mask_features = self.mask_pooler(features, pred_boxes)
            mask_logits = self.mask_head(mask_features)
            mask_rcnn_inference(mask_logits, instances)
            return instances

    def _forward_keypoint(self, inputs,features, instances):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (list[Tensor]): #level input features for keypoint prediction
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.is_training else instances

        raise NotImplementedError()


