#coding=utf-8
from .roi_heads import *
from object_detection2.modeling.poolers import ROIPooler
import basic_tftools as btf
from object_detection2.datadef import *
from .mask_head import *
from .build import *
from object_detection2.odtools import *
from .box_head import BoxesForwardType
import wnnlayer as wnnl
import object_detection2.bboxes as odb

@ROI_HEADS_REGISTRY.register()
class BoxFreeROIHeads(ROIHeads):
    """
        It's "standard" in a sense that there is no ROI transform sharing
        or feature sharing between tasks.
        The cropped rois go to separate branches (boxes and masks) directly.
        This way, it is easier to make separate abstractions for different branches.

        This class is used by most models, such as FPN and C5.
        To implement more models, you can subclass it and implement a different
        :meth:`forward()` or a head.
        """

    def __init__(self, cfg, parent, *args, **kwargs):
        super().__init__(cfg, *args, parent=parent, **kwargs)
        self._init_box_head(cfg, *args, **kwargs)
        self._init_mask_head(cfg, *args, **kwargs)
        self.rcnn_outboxes = None
        self.test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img = cfg.TEST.DETECTIONS_PER_IMAGE
        self.box_scale = 2.0

    def _init_box_head(self, cfg, *args, **kwargs):

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        bin_size = cfg.MODEL.ROI_BOX_HEAD.bin_size
        '''
        是否将RCNN预测的结果放mask/keypoint 在分支输入的proposal box中，Detectron2所有的默认设置及配置文件中这一项都是False
        '''
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same

        self.box_pooler = ROIPooler(cfg=cfg.MODEL.ROI_BOX_HEAD, parent=self,
                                    output_size=pooler_resolution,
                                    bin_size=bin_size,
                                    pooler_type=pooler_type,
                                    )

    def _init_mask_head(self, cfg, *args, **kwargs):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        bin_size = cfg.MODEL.ROI_MASK_HEAD.bin_size
        # fmt: on

        self.mask_pooler = ROIPooler(cfg=cfg.MODEL.ROI_MASK_HEAD, parent=self,
                                     output_size=pooler_resolution,
                                     bin_size=bin_size,
                                     pooler_type=pooler_type,
                                     )
        self.mask_head = build_mask_head(
            cfg,
            parent=self,
            **kwargs
        )


    def forward(self, inputs, features, proposals: ProposalsData):
        """
        See :class:`ROIHeads.forward`.
        """
        self.batched_inputs = inputs
        proposals_boxes = proposals[PD_BOXES]
        if self.is_training:
            proposals = self.label_and_sample_proposals(inputs, proposals_boxes)

        features_list = [features[f] for f in self.in_features]

        img_size = get_img_size_from_batched_inputs(inputs)
        if self.is_training:
            pred_instances, losses = self._forward_box(features_list, proposals, img_size=img_size)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            if self.train_on_pred_boxes:
                # proposals里面的box已经是采样的结果,无需再进行采样操作
                proposals = self.label_and_sample_proposals(inputs, proposals.boxes, do_sample=False)
            mk_loss,mk_instance = self._forward_mask(inputs, features_list, proposals, img_size=img_size)
            losses.update(mk_loss)
            pred_instances.update(mk_instance)
            return pred_instances, losses
        else:
            pred_instances, _ = self._forward_box(features_list, proposals, img_size=img_size)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            mk_pred_instances = self.forward_with_given_boxes(inputs, features, pred_instances, img_size=img_size)
            pred_instances.update(mk_pred_instances)
            if self.cfg.GLOBAL.SUMMARY_LEVEL <= SummaryLevel.RESEARCH and not self.is_training:
                matcher = Matcher(
                    [1e-3],
                    allow_low_quality_matches=False,
                    cfg=self.cfg,
                    parent=self
                )
                proposal_boxes = wmlt.batch_gather(proposals[PD_BOXES],pred_instances[RD_INDICES])
                mh_res0 = matcher(proposal_boxes,
                                  self.batched_inputs[GT_BOXES],
                                  self.batched_inputs[GT_LABELS],
                                  self.batched_inputs[GT_LENGTH])

                with tf.device("/cpu:0"):

                    lens = pred_instances[RD_LENGTH]
                    boxes = pred_instances[RD_BOXES]
                    scores0 = mh_res0[1]
                    l = lens[0]
                    probability = pred_instances[RD_PROBABILITY]
                    mh_res1 = matcher(boxes[:, :l],
                                      self.batched_inputs[GT_BOXES],
                                      self.batched_inputs[GT_LABELS],
                                      self.batched_inputs[GT_LENGTH])
                    add_to_research_datas("rd_scores", mh_res1[1][:, :l], [-1])

                    add_to_research_datas("rd_probs", probability[:, :l], [-1])
                    add_to_research_datas("rd_probs", probability[:, :l], [-1])
                    add_to_research_datas("rd_scores_old", scores0[:, :l], [-1])

            return pred_instances, {}

    def forward_with_given_boxes(self, inputs, features, instances, img_size):
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

        instances = self._forward_mask(inputs, features, instances, img_size=img_size)
        return instances

    def _forward_box(self, features, proposals, img_size, retry=True):
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
            proposal_boxes = proposals.boxes  # when training proposals's EncodedData
        else:
            proposal_boxes = proposals[PD_BOXES]  # when inference proposals's a dict which is the outputs of RPN
        box_features = self.box_pooler(features, proposal_boxes, img_size=img_size)
        if self.roi_hook is not None:
            box_features = self.roi_hook(box_features, self.batched_inputs)
        pred_class_logits = self.pred_classes(box_features)

        outputs = build_outputs(name=self.cfg.MODEL.ROI_HEADS.OUTPUTS,
                                cfg=self.cfg, parent=self,
                                box2box_transform=self.box2box_transform,
                                pred_class_logits=pred_class_logits,
                                pred_proposal_deltas=None,
                                pred_iou_logits=None,
                                proposals=proposals,
                                )
        if self.is_training:
            if self.train_on_pred_boxes:
                pred_boxes = outputs.predict_boxes_for_gt_classes()
                self.rcnn_outboxes = pred_boxes
                pred_instances = {}
            return {}, outputs.losses()
        else:
            pred_instances = outputs.inference(
                self.test_score_thresh,
                proposal_boxes=tf.reshape(proposal_boxes,shape=[-1,4]),
            )

            return pred_instances, {}

    def _forward_mask(self, inputs, features, instances, img_size):
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

        if self.is_training:
            # when training, instance is EncodedData
            # The loss is only defined on positive proposals.
            fg_selection_mask = select_foreground_proposals(instances)
            t_bboxes = odb.scale_bboxes(instances.boxes,(self.box_scale,self.box_scale),correct=True)
            B,Nr,_ = wmlt.combined_static_and_dynamic_shape(t_bboxes)
            bboxes = tf.stack([t_bboxes,instances.boxes],axis=2)
            index = tf.random_uniform(shape=[B,Nr,1],minval=0,maxval=2,dtype=tf.int32)
            bboxes = wmlt.batch_gather(bboxes,index)
            bboxes = tf.squeeze(bboxes,axis=2)
            instances.boxes = bboxes
            proposal_boxes = instances.boxes
            mask_features = self.mask_pooler(features, proposal_boxes, img_size=img_size)
            fg_selection_mask = tf.reshape(fg_selection_mask, [-1])
            mask_features = tf.boolean_mask(mask_features, fg_selection_mask)
            mask_logits = self.mask_head(mask_features)
            return {"loss_mask": mask_rcnn_loss(inputs, mask_logits, instances, fg_selection_mask)},{}
        else:
            # when inference instances is RCNNResultsData
            instances[RD_BOXES] = odb.scale_bboxes(instances[RD_BOXES],(self.box_scale,self.box_scale),correct=True)
            pred_boxes = instances[RD_BOXES]
            mask_features = self.mask_pooler(features, pred_boxes, img_size=img_size)
            mask_logits = self.mask_head(mask_features)
            mask_rcnn_inference(mask_logits, instances)
            self.merge_bboxes_and_mask_results(instances)
            return instances

    @wmlt.add_name_scope
    def merge_bboxes_and_mask_results(self,instances):
        
        nms = functools.partial(tfop.boxes_nms, threshold=self.test_nms_thresh, classes_wise=True)
        mask = tf.squeeze(instances[RD_MASKS],axis=0)
        bmask = tf.expand_dims(mask,axis=-1)
        bmask = wnnl.min_pool2d(bmask,kernel_size=2,stride=1,padding="SAME")
        bmask = slim.max_pool2d(bmask,kernel_size=2,stride=1,padding="SAME")
        bmask = tf.squeeze(bmask,axis=-1)
        bmask = tf.cast(bmask*255,tf.uint8)
        #bboxes = tf.squeeze(instances[RD_BOXES],axis=0)
        bboxes = tfop.get_bboxes_from_mask(bmask)
        #bboxes = tf.Print(bboxes,["xbboxes",tf.reduce_max(xboxes,keepdims=False),
                                  #tf.reduce_max(mask)],summarize=100)
        Nr,H,W = btf.combined_static_and_dynamic_shape(mask)
        bboxes = odb.tfabsolutely_boxes_to_relative_boxes(bboxes,width=W,height=H)
        mask = tf.expand_dims(mask,axis=-1)
        mask = wmlt.tf_crop_and_resize(mask,bboxes,size=(H,W))
        mask = tf.squeeze(mask,axis=-1)
        bboxes = odb.restore_sub_area(bboxes,sub_box=tf.squeeze(instances[RD_BOXES],axis=0))

        probability = tf.squeeze(instances[RD_PROBABILITY],axis=0)
        labels = tf.squeeze(instances[RD_LABELS],axis=0)
        res_indices = tf.squeeze(instances[RD_INDICES],axis=0)
        bboxes_area = odb.box_area(bboxes)
        keep_mask = tf.greater(bboxes_area,1e-3)
        bboxes = tf.boolean_mask(bboxes,keep_mask)
        labels = tf.boolean_mask(labels,keep_mask)
        probability = tf.boolean_mask(probability,keep_mask)
        res_indices = tf.boolean_mask(res_indices,keep_mask)
        mask = tf.boolean_mask(mask,keep_mask)

        bboxes, labels, indices = nms(bboxes, labels, confidence=probability)
        probability = tf.gather(probability, indices)
        res_indices = tf.gather(res_indices, indices)
        mask = tf.gather(mask,indices)

        instances[RD_BOXES] = tf.expand_dims(bboxes,axis=0)
        instances[RD_MASKS] = tf.expand_dims(mask,axis=0)
        instances[RD_LABELS] = tf.expand_dims(labels,axis=0)
        instances[RD_PROBABILITY] = tf.expand_dims(probability,axis=0)
        instances[RD_INDICES] = tf.expand_dims(res_indices,axis=0)
        instances[RD_LENGTH] = tf.shape(labels)


    @staticmethod
    def get_boxes_from_mask(mask):
        '''

        :param mask: [B,Y,mask_H,mask_W]
        :return:
        '''

    def pred_classes(self, x, scope="PredClasses", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            fc_dim = self.cfg.MODEL.ROI_BOX_HEAD.FC_DIM
            num_fc = self.cfg.MODEL.ROI_BOX_HEAD.NUM_FC
            normalizer_fn, norm_params = odt.get_norm(self.cfg.MODEL.ROI_BOX_HEAD.NORM, self.is_training)
            activation_fn = odt.get_activation_fn(self.cfg.MODEL.ROI_BOX_HEAD.ACTIVATION_FN)

            cls_x = x
            with tf.variable_scope("ClassPredictionTower"):
                if num_fc > 0:
                    if len(cls_x.get_shape()) > 2:
                        shape = wmlt.combined_static_and_dynamic_shape(cls_x)
                        dim = 1
                        for i in range(1, len(shape)):
                            dim = dim * shape[i]
                        cls_x = tf.reshape(cls_x, [shape[0], dim])
                    for _ in range(num_fc):
                        cls_x = slim.fully_connected(cls_x, fc_dim,
                                                     activation_fn=activation_fn,
                                                     normalizer_fn=normalizer_fn,
                                                     normalizer_params=norm_params)
            scores = slim.fully_connected(cls_x, self.num_classes + 1, activation_fn=None,
                                          normalizer_fn=None, scope="cls_score")
            return scores
