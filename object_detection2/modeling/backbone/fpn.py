import math
import tensorflow as tf
import wmodule
import functools
from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from .resnet import build_resnet_backbone
from .shufflenetv2 import build_shufflenetv2_backbone
import collections
import object_detection2.od_toolkit as odt
from .build import build_backbone_hook

slim = tf.contrib.slim

class FPN(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, cfg,bottom_up, in_features, out_channels, top_block=None, fuse_type="sum",
            parent=None,*args,**kwargs
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
        super(FPN, self).__init__(cfg,parent=parent,*args,**kwargs)
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
        #Detectron2默认没有使用normalizer, 但在测试数据集上发现不使用normalizer网络不收敛
        self.normalizer_fn,self.norm_params = odt.get_norm(self.cfg.MODEL.FPN.NORM,self.is_training)
        self.hook_before,self.hook_after = build_backbone_hook(cfg.MODEL.FPN,parent=self)


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
        if self.hook_before is not None:
            bottom_up_features = self.hook_before(bottom_up_features,x)
        image_features = [bottom_up_features[f] for f in self.in_features]
        use_depthwise = self.use_depthwise
        depth = self.out_channels

        with tf.variable_scope(self.scope, 'top_down'):
            num_levels = len(image_features)
            output_feature_maps_list = []
            output_feature_map_keys = []
            padding = 'SAME'
            kernel_size = 3
            weight_decay = 1e-4
            if self.normalizer_fn is not None:
                normalizer_fn = functools.partial(self.normalizer_fn,**self.norm_params)
            else:
                normalizer_fn = None
            if use_depthwise:
                conv_op = functools.partial(slim.separable_conv2d, depth_multiplier=1,
                normalizer_fn=normalizer_fn)
            else:
                conv_op = functools.partial(slim.conv2d,
                weights_regularizer=slim.l2_regularizer(weight_decay),
                normalizer_fn=normalizer_fn)
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
        res = collections.OrderedDict(zip(output_feature_map_keys, output_feature_maps_list))
        if self.hook_after is not None:
            res = self.hook_after(res,x)
        return res

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

    def __init__(self, out_channels,cfg,*args,**kwargs):
        super().__init__(cfg=cfg,*args,**kwargs)
        self.normalizer_fn,self.norm_params = odt.get_norm(self.cfg.MODEL.FPN.NORM,self.is_training)
        self.out_channels = out_channels

    def forward(self, c5):
        with tf.variable_scope("FPNLastLevel"):
            res = []
            last_feature = c5
            for i in range(self.cfg.MODEL.FPN.LAST_LEVEL_NUM_CONV):
                last_feature  = slim.conv2d(last_feature,self.out_channels,[3,3],stride=2,activation_fn=tf.nn.relu,
                         normalizer_fn=self.normalizer_fn,normalizer_params=self.norm_params,
                         scope=f"conv{i+1}")
                res.append(last_feature)
        return res


@BACKBONE_REGISTRY.register()
def build_resnet_fpn_backbone(cfg,*args,**kwargs):
    """
    Args:
        cfg: a CfgNode

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
        cfg: a CfgNode

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
        cfg: a CfgNode

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
        cfg: a CfgNode

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
