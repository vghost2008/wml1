#coding=utf-8
import tensorflow as tf
import wmodule
from .build import META_ARCH_REGISTRY
from object_detection2.modeling.build import build_outputs
from object_detection2.modeling.backbone.build import build_backbone
from object_detection2.modeling.anchor_generator import build_anchor_generator
from object_detection2.modeling.box_regression import FCOSBox2BoxTransform
from object_detection2.modeling.matcher import Matcher
import math
from object_detection2.standard_names import *
from object_detection2.modeling.onestage_heads.retinanet_outputs import *
from .meta_arch import MetaArch
from object_detection2.datadef import *
import object_detection2.od_toolkit as odtk

slim = tf.contrib.slim

__all__ = ["FCOS"]



@META_ARCH_REGISTRY.register()
class FCOS(MetaArch):
    """
    Implement FCOS: Fully Convolutional One-Stage Object Detection
    """

    def __init__(self, cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)

        # fmt: off
        self.num_classes              = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features              = cfg.MODEL.FCOS.IN_FEATURES
        # fmt: on

        self.backbone = build_backbone(cfg,parent=self,*args,**kwargs)

        self.head = FCOSHead(cfg=cfg.MODEL.FCOS,
                             parent=self,
                             *args,**kwargs)

        # Matching and loss
        self.box2box_transform = FCOSBox2BoxTransform()

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
        if len(self.in_features) == 0:
            print(f"Error no input features for retinanet, use all features {features.keys()}")
            features = list(features.values())
        else:
            features = [features[f] for f in self.in_features]
        pred_logits, pred_regression,pred_center_ness = self.head(features)
        gt_boxes = batched_inputs[GT_BOXES]
        gt_length = batched_inputs[GT_LENGTH]
        gt_labels = batched_inputs[GT_LABELS]

        outputs = build_outputs(name=self.cfg.MODEL.FCOS.OUTPUTS,
            cfg=self.cfg.MODEL.FCOS,
            parent=self,
            box2box_transform=self.box2box_transform,
            pred_logits=pred_logits,
            pred_regression =pred_regression,
            pred_center_ness=pred_center_ness,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_length=gt_length,
            batched_inputs=batched_inputs,
            max_detections_per_image=self.cfg.TEST.DETECTIONS_PER_IMAGE,
        )

        if self.is_training:
            if self.cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
                results = outputs.inference(inputs=batched_inputs,box_cls=pred_logits,
                                            box_regression=pred_regression,
                                            center_ness=pred_center_ness)
            else:
                results = {}

            return results,outputs.losses()
        else:
            results = outputs.inference(inputs=batched_inputs,box_cls=pred_logits,
                                        box_regression=pred_regression,
                                        center_ness=pred_center_ness)
            return results,{}


class FCOSHead(wmodule.WChildModule):
    """
    The head used in FCOS for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg,parent,*args,**kwargs):
        '''

        :param cfg:  only the child part
        :param parent:
        :param args:
        :param kwargs:
        '''
        super().__init__(cfg,*args,parent=parent,**kwargs)
        self.normalizer_fn,self.norm_params = odtk.get_norm(self.cfg.NORM,is_training=self.is_training)
        self.activation_fn = odtk.get_activation_fn(self.cfg.ACTIVATION_FN)
        self.norm_scope_name = odtk.get_norm_scope_name(self.cfg.NORM)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, Hi, Wi,AxK).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Hi, Wi, Ax4).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        cfg = self.cfg
        num_classes      = cfg.NUM_CLASSES
        num_convs        = cfg.NUM_CONVS
        prior_prob       = cfg.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        logits = []
        bbox_reg = []
        center_ness = []
        for j,feature in enumerate(features):
            channels = feature.get_shape().as_list()[-1]
            with tf.variable_scope("WeightSharedConvolutionalBoxPredictor", reuse=tf.AUTO_REUSE):
                net = feature
                with tf.variable_scope("BoxPredictionTower"):
                    for i in range(num_convs):
                        net = slim.conv2d(net,channels,[3,3],
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          biases_initializer=None if self.normalizer_fn is not None else tf.zeros_initializer(),
                                          scope=f"conv2d_{i}")
                        if self.normalizer_fn is not None:
                            with tf.variable_scope(f"conv2d_{i}"):
                                net = self.normalizer_fn(net, scope=f'{self.norm_scope_name}/feature_{j}',**self.norm_params)
                        if self.activation_fn is not None:
                            net = self.activation_fn(net)
                _bbox_reg = slim.conv2d(net, 4, [3, 3], activation_fn=tf.math.exp,
                                         normalizer_fn=None,
                                         scope="BoxPredictor")*tf.get_variable(name=f"gamma_{j}",shape=(),initializer=tf.ones_initializer())*math.exp(j)
                net = feature
                with tf.variable_scope("ClassPredictionTower"):
                    for i in range(num_convs):
                        net = slim.conv2d(net,channels,[3,3],
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          biases_initializer=None if self.normalizer_fn is not None else tf.zeros_initializer(),
                                          scope=f"conv2d_{i}")
                        if self.normalizer_fn is not None:
                            with tf.variable_scope(f"conv2d_{i}"):
                                net = self.normalizer_fn(net, scope=f'{self.norm_scope_name}/feature_{j}',**self.norm_params)
                        if self.activation_fn is not None:
                            net = self.activation_fn(net)
                _logits = slim.conv2d(net, num_classes, [3, 3], activation_fn=None,
                                         normalizer_fn=None,
                                         biases_initializer=tf.constant_initializer(value=bias_value),
                                         scope="ClassPredictor")
                _center_ness = slim.conv2d(net, 1, [3, 3], activation_fn=None,
                                        normalizer_fn=None,
                                        scope="CenterNessPredictor")
                _center_ness = tf.squeeze(_center_ness,axis=-1)

            logits.append(_logits)
            bbox_reg.append(_bbox_reg)
            center_ness.append(_center_ness)
        return logits, bbox_reg,center_ness
