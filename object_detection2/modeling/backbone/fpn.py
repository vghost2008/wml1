import math
import tensorflow as tf
import wmodule
import functools
from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from .resnet import build_resnet_backbone
from .shufflenetv2 import build_shufflenetv2_backbone
import collections

slim = tf.contrib.slim

class FPN(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, cfg,bottom_up, in_features, out_channels, top_block=None, fuse_type="sum",
            *args,**kwargs
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        stage = int(in_features[-1][1:])
        super(FPN, self).__init__(cfg,*args,**kwargs)
        assert isinstance(bottom_up, Backbone)

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        self.out_channels = out_channels
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type
        self.scope = "FPN"
        self.use_depthwise = False
        self.interpolate_op=tf.image.resize_nearest_neighbor
        self.stage = stage

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        bottom_up_features = self.bottom_up(x)
        image_features = [bottom_up_features[f] for f in self.in_features]
        use_depthwise = self.use_depthwise
        depth = self.out_channels

        with tf.variable_scope(self.scope, 'top_down'):
            num_levels = len(image_features)
            output_feature_maps_list = []
            output_feature_map_keys = []
            padding = 'SAME'
            kernel_size = 3
            if use_depthwise:
                conv_op = functools.partial(slim.separable_conv2d, depth_multiplier=1)
            else:
                conv_op = slim.conv2d
            with slim.arg_scope(
                    [slim.conv2d], padding=padding, stride=1):
                prev_features = slim.conv2d(
                    image_features[-1],
                    depth, [1, 1], activation_fn=None, normalizer_fn=None,
                    scope='projection_%d' % num_levels)
                output = conv_op(prev_features, depth,[kernel_size, kernel_size], scope=f"output_{num_levels}")
                output_feature_maps_list.append(output)
                output_feature_map_keys.append(f"P{self.stage}")

                for level in reversed(range(num_levels - 1)):
                    lateral_features = slim.conv2d(
                        image_features[level], depth, [1, 1],
                        activation_fn=None, normalizer_fn=None,
                        scope='projection_%d' % (level + 1))
                    shape = tf.shape(lateral_features)[1:3]
                    top_down = self.interpolate_op(prev_features, shape)
                    prev_features = top_down + lateral_features
                    output_feature_maps_list.append(conv_op(
                        prev_features,
                        depth, [kernel_size, kernel_size],
                        scope=f'output_{level + 1}'))
                    output_feature_map_keys.append(f"P{self.stage+level-num_levels+1}")
        output_feature_map_keys.reverse()
        output_feature_maps_list.reverse()
        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.in_features[-1], None)
            if top_block_in_feature is not None:
                res = self.top_block(top_block_in_feature)
                output_feature_maps_list.extend(res)
                for i in range(len(res)):
                    output_feature_map_keys.append(f"P{self.stage+i+1}")
        return collections.OrderedDict(zip(output_feature_map_keys, output_feature_maps_list))

class LastLevelMaxPool(wmodule.WChildModule):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.num_levels = 1

    def forward(self, x):
        return [slim.max_pool2d(x, kernel_size=1, stride=2, padding="SAME")]


class LastLevelP6P7(wmodule.WChildModule):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, out_channels,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.out_channels = out_channels

    def forward(self, c5):
        p6 = slim.conv2d(c5,self.out_channels,[3,3],stride=2,activation_fn=None,
                         normalizer_fn=None)
        p7 = slim.conv2d(tf.nn.relu(p6),self.out_channels,[3,3],stride=2,activation_fn=None,normalizer_fn=None)
        return [p6, p7]


@BACKBONE_REGISTRY.register()
def build_resnet_fpn_backbone(cfg,*args,**kwargs):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg,*args,**kwargs)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        top_block=LastLevelMaxPool(cfg,*args,**kwargs),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        cfg=cfg,
        *args,
        **kwargs
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_retinanet_resnet_fpn_backbone(cfg, *args,**kwargs):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, *args,**kwargs)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(out_channels,cfg,*args,**kwargs),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        cfg=cfg,
        *args,
        **kwargs
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_retinanet_shufflenetv2_fpn_backbone(cfg, *args,**kwargs):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_shufflenetv2_backbone(cfg, *args,**kwargs)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        top_block=LastLevelP6P7(out_channels,cfg,*args,**kwargs),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        cfg=cfg,
        *args,
        **kwargs
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_shufflenetv2_fpn_backbone(cfg,*args,**kwargs):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_shufflenetv2_backbone(cfg,*args,**kwargs)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        top_block=LastLevelMaxPool(cfg,*args,**kwargs),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        cfg=cfg,
        *args,
        **kwargs
    )
    return backbone
