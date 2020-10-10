import logging
import wmodule
import wml_tfutils as wmlt
from object_detection2.modeling.backbone.build import build_backbone,build_backbone_by_name
from object_detection2.modeling.bbdnet.build import build_bbdnet
from object_detection2.modeling.proposal_generator.build import build_proposal_generator
from .build import META_ARCH_REGISTRY
from object_detection2.datadef import *
from object_detection2.config.config import global_cfg
from object_detection2.modeling.roi_heads.roi_heads import build_roi_heads
import wsummary
from object_detection2.standard_names import *
import numpy as np
import tensorflow as tf
from .meta_arch import MetaArch
import img_utils as wmli
import cv2
import wml_utils as wmlu
import image_visualization as ivs
import basic_tftools as btf
from .meta_arch import MetaArch
from object_detection2.data.dataloader import DataLoader

@META_ARCH_REGISTRY.register()
class GeneralizedGRCNN(MetaArch):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg,*args,parent=None,**kwargs):
        super().__init__(cfg,*args,parent=parent,**kwargs)

        self.backbone = build_backbone(cfg,parent=self,*args,**kwargs)
        self.proposal_generator = build_proposal_generator(cfg,parent=self,*args,**kwargs)
        self.roi_heads = build_roi_heads(cfg,parent=self,*args,**kwargs)
        self.roi_heads.disable_training()
        self.proposal_generator.disable_training()
        self.roi_heads_backbone = None
        if cfg.MODEL.ROI_HEADS.BACKBONE != "":
            self.roi_heads_backbone = build_backbone_by_name(cfg.MODEL.ROI_HEADS.BACKBONE,cfg,parent=self,*args,**kwargs)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (B,H, W,C) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.is_training:
            return self.inference(batched_inputs)

        batched_inputs = self.preprocess_image(batched_inputs)

        '''
        使用主干网络生成一个FeatureMap, 如ResNet的Res4(stride=16)
        '''
        features = self.backbone(batched_inputs)
        bb_features = [features[f] for f in self.cfg.MODEL.ROI_HEADS.IN_FEATURES]
        if self.roi_heads_backbone is not None:
            roi_features = self.roi_heads_backbone(batched_inputs)
        else:
            roi_features = features

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(batched_inputs, features)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = {"proposal_boxes":batched_inputs["proposals"]}
            proposal_losses = {}

        results, detector_losses = self.roi_heads(batched_inputs, roi_features, proposals)

        print("Use crop map attr.")
        bbd_net_input = {}
        #bbd_net_input['net_data'] = map_data
        bbd_net_input['base_net'] = bb_features[-1]
        bbd_net_input[IMAGE] = batched_inputs[IMAGE]
        bbd_net_input['net_data'] = bb_features
        bbd_net = build_bbdnet(self.cfg.MODEL.BBDNET.NAME,
                               num_classes=self.cfg.MODEL.RETINANET.NUM_CLASSES,cfg=self.cfg,parent=self,
                               threshold=0.02)
        if len(results)>0:
            wsummary.detection_image_summary(images=batched_inputs[IMAGE],
                                             boxes=results[RD_BOXES], classes=results[RD_LABELS],
                                             lengths=results[RD_LENGTH],
                                             scores=results[RD_PROBABILITY],
                                             name="RCNN_result",
                                             category_index=DataLoader.category_index)

        losses = {}
        if self.is_training:
            bbd_net_input.update(results)
            bbd_net_input[GT_BOXES] = batched_inputs[GT_BOXES]
            bbd_net_input[GT_LABELS] = batched_inputs[GT_LABELS]
            bbd_net_input[GT_LENGTH] = batched_inputs[GT_LENGTH]
            ###
            bbd_net_input[RD_RAW_PROBABILITY] = results[RD_PROBABILITY]
            ###
            bbd_loss,bbd_outputs = bbd_net(bbd_net_input)
            results.update(bbd_outputs)
            losses.update(bbd_loss)

        else:
            bbd_net_input.update(results)
            ###
            bbd_net_input[RD_RAW_PROBABILITY] = results[RD_PROBABILITY]
            ###
            bbd_loss,bbd_outputs = bbd_net(bbd_net_input)
            results.update(bbd_outputs)

        if global_cfg.GLOBAL.SUMMARY_LEVEL <= SummaryLevel.DEBUG:
            wsummary.detection_image_summary(images=batched_inputs[IMAGE],
                                             boxes=results[RD_BOXES], classes=results[RD_LABELS],
                                             lengths=results[RD_LENGTH],
                                             scores=results[RD_PROBABILITY],
                                             name="GRCNN_result",
                                             category_index=DataLoader.category_index)


        losses.update(detector_losses)
        losses.update(proposal_losses)
        return results,losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.is_training

        batched_inputs = self.preprocess_image(batched_inputs)
        features = self.backbone(batched_inputs)
        bb_features = [features[f] for f in self.cfg.MODEL.ROI_HEADS.IN_FEATURES]
        if self.roi_heads_backbone is not None:
            roi_features = self.roi_heads_backbone(batched_inputs)
        else:
            roi_features = features

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(batched_inputs, features)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(batched_inputs, roi_features, proposals)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(roi_features, detected_instances)

        instance_masks = None if not self.cfg.MODEL.MASK_ON else results.get(RD_MASKS,None)
        print("Use crop map attr.")
        bbd_net_input = {}
        #bbd_net_input['net_data'] = map_data
        bbd_net_input['base_net'] = bb_features[-1]
        bbd_net_input[IMAGE] = batched_inputs[IMAGE]
        bbd_net_input['net_data'] = bb_features
        bbd_net = build_bbdnet(self.cfg.MODEL.BBDNET.NAME,
                               num_classes=self.cfg.MODEL.RETINANET.NUM_CLASSES,cfg=self.cfg,parent=self,
                               threshold=0.02)

        losses = {}
        if self.is_training:
            bbd_net_input.update(results)
            bbd_net_input[GT_BOXES] = batched_inputs[GT_BOXES]
            bbd_net_input[GT_LABELS] = batched_inputs[GT_LABELS]
            bbd_net_input[GT_LENGTH] = batched_inputs[GT_LENGTH]
            ###
            bbd_net_input[RD_RAW_PROBABILITY] = results[RD_PROBABILITY]
            ###
            bbd_loss,bbd_outputs = bbd_net(bbd_net_input)
            results.update(bbd_outputs)
            losses.update(bbd_loss)

        else:
            bbd_net_input.update(results)
            ###
            bbd_net_input[RD_RAW_PROBABILITY] = results[RD_PROBABILITY]
            ###
            bbd_loss,bbd_outputs = bbd_net(bbd_net_input)
            results.update(bbd_outputs)

        if global_cfg.GLOBAL.SUMMARY_LEVEL <= SummaryLevel.DEBUG:
            wsummary.detection_image_summary(images=batched_inputs[IMAGE],
                                             boxes=results[RD_BOXES], classes=results[RD_LABELS],
                                             lengths=results[RD_LENGTH],
                                             scores=results[RD_PROBABILITY],
                                             name="GRCNN_result",
                                             category_index=DataLoader.category_index)

        if instance_masks is not None:
            shape = btf.combined_static_and_dynamic_shape(batched_inputs[IMAGE])
            instance_masks = tf.cast(instance_masks>0.5,tf.float32)
            instance_masks = ivs.batch_tf_get_fullsize_mask(boxes=results[RD_BOXES],
                                                   masks=instance_masks,
                                                   size=shape[1:3]
                                                   )
        wsummary.detection_image_summary(images=batched_inputs[IMAGE],
                                         boxes=results[RD_BOXES],classes=results[RD_LABELS],
                                         lengths=results[RD_LENGTH],
                                         scores=results[RD_PROBABILITY],
                                         instance_masks=instance_masks,name="RCNN_result",
                                         category_index=DataLoader.category_index)

        if instance_masks is not None:
            wsummary.detection_image_summary(images=tf.zeros_like(batched_inputs[IMAGE]),
                                             boxes=results[RD_BOXES],classes=results[RD_LABELS],
                                             lengths=results[RD_LENGTH],
                                             instance_masks=instance_masks,
                                             name="RCNN_Mask_result",
                                             category_index=DataLoader.category_index)

        if do_postprocess:
            return self._postprocess(results, batched_inputs),None
        else:
            return results,None
