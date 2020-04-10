#coding=utf-8
import tensorflow as tf
import wmodule
from .build import META_ARCH_REGISTRY,build_outputs
from object_detection2.modeling.backbone.build import build_backbone
from object_detection2.modeling.anchor_generator import build_anchor_generator
from object_detection2.modeling.box_regression import Box2BoxTransform
from object_detection2.modeling.matcher import Matcher
import math
from object_detection2.standard_names import *
from object_detection2.modeling.onestage_heads.retinanet_outputs import *
from .meta_arch import MetaArch
from object_detection2.datadef import *

slim = tf.contrib.slim

__all__ = ["YOLACT"]



@META_ARCH_REGISTRY.register()
class YOLACT(MetaArch):
    """
    Implement YOLACT++ Better Real-time Instance Segmentation
    """

    def __init__(self, cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)

        # fmt: off
        self.num_classes              = cfg.MODEL.YOLACT.NUM_CLASSES
        self.in_features              = cfg.MODEL.YOLACT.IN_FEATURES
        # fmt: on

        self.backbone = build_backbone(cfg,parent=self,*args,**kwargs)

        self.anchor_generator = build_anchor_generator(cfg,parent=self,*args,**kwargs)
        self.head = YOLACTHead(cfg=cfg.MODEL.YOLACT,
                               num_anchors=self.anchor_generator.num_cell_anchors,
                               parent=self,
                               coefficient_nr=self.cfg.MODEL.YOLACT.PROTONET_NR,
                               *args,**kwargs)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.YOLACT.IOU_THRESHOLDS,
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
        head_outputs = self.head(features)
        anchors = self.anchor_generator(batched_inputs,features)
        gt_boxes = batched_inputs[GT_BOXES]
        gt_length = batched_inputs[GT_LENGTH]
        gt_labels = batched_inputs[GT_LABELS]

        outputs = build_outputs(name=self.cfg.MODEL.YOLACT.OUTPUTS,
            cfg=self.cfg.MODEL.YOLACT,
            parent=self,
            box2box_transform=self.box2box_transform,
            anchor_matcher=self.anchor_matcher,
            pred_logits=head_outputs[LOGITS],
            pred_anchor_deltas=head_outputs[BOXES_REGS],
            anchors=anchors,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_length=gt_length,
            max_detections_per_image=self.cfg.TEST.DETECTIONS_PER_IMAGE,
            head_outputs=head_outputs,
            batched_inputs=batched_inputs,
            coefficient_nr=self.cfg.MODEL.YOLACT.PROTONET_NR,

        )

        if self.is_training:
            if self.cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
                results = outputs.inference(inputs=batched_inputs,box_cls=head_outputs[LOGITS],
                                            box_delta=head_outputs[BOXES_REGS], anchors=anchors)
            else:
                results = {}

            return results,outputs.losses()
        else:
            results = outputs.inference(inputs=batched_inputs,box_cls=head_outputs[LOGITS],
                                        box_delta=head_outputs[BOXES_REGS], anchors=anchors)
            return results,{}


class YOLACTHead(wmodule.WChildModule):
    """
    The head used in YOLACT for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, num_anchors,coefficient_nr,cfg,parent,*args,**kwargs):
        '''

        :param num_anchors:
        :param cfg:  only the child part
        :param parent:
        :param args:
        :param kwargs:
        '''
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
        # Detectron2默认没有使用normalizer, 但在测试数据集上发现不使用normalizer网络不收敛
        self.normalizer_fn = slim.batch_norm
        self.coefficient_nr = coefficient_nr
        self.activation_fn = tf.nn.relu

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
        coeffs = []
        outdata = {}
        with tf.variable_scope("Protonet"):
            net = features[0]
            channels = net.get_shape().as_list()[-1]
            for i in range(num_convs):
                net = slim.conv2d(net, channels, [3, 3],
                                  activation_fn=self.activation_fn,
                                  normalizer_fn=self.normalizer_fn,
                                  normalizer_params=self.norm_params,
                                  scope=f"conv2d_{i}")
        protos = slim.conv2d(net, self.coefficient_nr, [3, 3], activation_fn=tf.nn.relu,
                            normalizer_fn=None,
                            scope="ProtoPredictor")
        outdata["protos"] = protos

        if self.is_training:
            with tf.variable_scope("Semantic"):
                net = slim.conv2d(net, channels, [3, 3],
                                  activation_fn=self.activation_fn,
                                  normalizer_fn=self.normalizer_fn,
                                  normalizer_params=self.norm_params,
                                  scope=f"conv2d_{0}")
            semantic = slim.conv2d(net, num_classes, [3, 3], activation_fn=None,
                                 normalizer_fn=None,
                                 scope="SemanticPredictor")
            outdata[SEMANTIC] = semantic

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
                _logits = slim.conv2d(net, self.num_anchors* num_classes, [3, 3], activation_fn=None,
                                         normalizer_fn=None,
                                         biases_initializer=tf.constant_initializer(value=bias_value),
                                         scope="ClassPredictor")

                with tf.variable_scope("CoefficientPredictionTower"):
                    net = feature
                    for i in range(num_convs):
                        net = slim.conv2d(net,channels,[3,3],
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          scope=f"conv2d_{i}")
                        if self.normalizer_fn is not None:
                            net = self.normalizer_fn(net, scope=f'conv2d_{i}/BatchNorm/feature_{j}',**self.norm_params)
                        net = self.activation_fn(net)
                _coeff = slim.conv2d(net, self.num_anchors* self.coefficient_nr, [3, 3], activation_fn=tf.nn.tanh,
                                      normalizer_fn=None,
                                      biases_initializer=tf.constant_initializer(value=bias_value),
                                      scope="CoefficientPredictor")

            logits.append(_logits)
            bbox_reg.append(_bbox_reg)
            coeffs.append(_coeff)
            outdata[LOGITS] = logits
            outdata[BOXES_REGS] = bbox_reg
            outdata[COEFFICIENT] = coeffs

        return outdata
