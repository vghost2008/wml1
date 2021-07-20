#coding=utf-8
import tensorflow as tf
import wmodule
from .build import META_ARCH_REGISTRY
from object_detection2.modeling.backbone.build import build_backbone
from object_detection2.modeling.anchor_generator import build_anchor_generator
from object_detection2.modeling.box_regression import Box2BoxTransform
from object_detection2.modeling.matcher import Matcher
import math
from object_detection2.standard_names import *
from object_detection2.modeling.onestage_heads.ssd_outputs import *
from .meta_arch import MetaArch
from object_detection2.datadef import *

slim = tf.contrib.slim

__all__ = ["SSD"]



@META_ARCH_REGISTRY.register()
class SSD(MetaArch):

    def __init__(self, cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)

        # fmt: off
        self.num_classes              = cfg.MODEL.SSD.NUM_CLASSES
        self.in_features              = cfg.MODEL.SSD.IN_FEATURES
        # Inference parameters:
        self.score_threshold          = cfg.MODEL.SSD.SCORE_THRESH_TEST
        self.topk_candidates          = cfg.MODEL.SSD.TOPK_CANDIDATES_TEST
        self.nms_threshold            = cfg.MODEL.SSD.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        self.batch_size               = cfg.SOLVER.IMS_PER_BATCH
        # fmt: on

        self.backbone = build_backbone(cfg,parent=self,*args,**kwargs)

        self.anchor_generator = build_anchor_generator(cfg,parent=self,*args,**kwargs)
        self.head = SSDHead(cfg=cfg, num_anchors=self.anchor_generator.num_cell_anchors,parent=self,
                                  *args,**kwargs)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.SSD.IOU_THRESHOLDS,
            allow_low_quality_matches=True,
            cfg=cfg,
            parent=self,
        )


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (H, W, C) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        batched_inputs = self.preprocess_image(batched_inputs)

        features = self.backbone(batched_inputs)
        features = [features[f] for f in self.in_features]
        pred_logits, pred_anchor_deltas= self.head(features)
        anchors = self.anchor_generator(batched_inputs,features)
        gt_boxes = batched_inputs[GT_BOXES]
        gt_length = batched_inputs[GT_LENGTH]
        gt_labels = batched_inputs[GT_LABELS]

        outputs = SSDOutputs(
            cfg=self.cfg,
            parent=self,
            box2box_transform=self.box2box_transform,
            anchor_matcher=self.anchor_matcher,
            pred_logits=pred_logits,
            pred_anchor_deltas=pred_anchor_deltas,
            anchors=anchors,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_length=gt_length
        )

        if self.is_training:
            if self.cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
                results = outputs.inference(inputs=batched_inputs, box_cls=pred_logits,
                                            box_delta=pred_anchor_deltas, anchors=anchors)
            else:
                results = {}

            return results, outputs.losses()
        else:
            results = outputs.inference(inputs=batched_inputs,box_cls=pred_logits,
                                        box_delta=pred_anchor_deltas, anchors=anchors)
            return results,{}

class SSDHead(wmodule.WChildModule):

    def __init__(self, num_anchors,cfg,parent,*args,**kwargs):
        super().__init__(cfg,*args,parent=parent,**kwargs)
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        self.num_anchors = num_anchors[0]
        self.norm_params = {
            'decay': 0.997,
            'epsilon': 1e-4,
            'scale': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'fused': None,  # Use fused batch norm if possible.
            'is_training': self.is_training
        }
        #测试时不使用batch_norm不收敛
        self.normalizer_fn = slim.batch_norm
        self.activation_fn = tf.nn.relu

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        cfg = self.cfg
        num_classes      = cfg.MODEL.SSD.NUM_CLASSES
        num_convs        = cfg.MODEL.SSD.NUM_CONVS
        prior_prob       = cfg.MODEL.SSD.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        logits = []
        bbox_reg = []
        for j,feature in enumerate(features):
            channels = feature.get_shape().as_list()[-1]
            with tf.variable_scope("WeightSharedConvolutionalBoxPredictor", reuse=tf.AUTO_REUSE):
                net = feature
                with tf.variable_scope("BoxPredictionTower"):
                    for i in range(num_convs):
                        net = slim.conv2d(net,channels,[3,3],
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          scope=f"conv2d_{i}")
                        if self.normalizer_fn is not None:
                            net = self.normalizer_fn(net, scope=f'conv2d_{i}/BatchNorm/feature_{j}',**self.norm_params)
                        net = self.activation_fn(net)
                _bbox_reg = slim.conv2d(net, self.num_anchors* 4, [3, 3], activation_fn=None,
                                         normalizer_fn=None,
                                         scope="BoxPredictor")
                net = feature
                with tf.variable_scope("ClassPredictionTower"):
                    for i in range(num_convs):
                        net = slim.conv2d(net,channels,[3,3],
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          scope=f"conv2d_{i}")
                        if self.normalizer_fn is not None:
                            net = self.normalizer_fn(net, scope=f'conv2d_{i}/BatchNorm/feature_{j}',**self.norm_params)
                        net = self.activation_fn(net)
                _logits = slim.conv2d(net, self.num_anchors* (num_classes+1), [3, 3], activation_fn=None,
                                         normalizer_fn=None,
                                         biases_initializer=tf.constant_initializer(value=bias_value),
                                         scope="ClassPredictor")

            logits.append(_logits)
            bbox_reg.append(_bbox_reg)
        return logits, bbox_reg
