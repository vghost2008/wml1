#coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
import collections

slim = tf.contrib.slim


def get_depth_fn(depth_multiplier, min_depth):
    """Builds a callable to compute depth (output channels) of conv filters.

    Args:
      depth_multiplier: a multiplier for the nominal depth.
      min_depth: a lower bound on the depth of filters.

    Returns:
      A callable that takes in a nominal depth and returns the depth to use.
    """

    def multiply_depth(depth):
        new_depth = int(depth * depth_multiplier)
        return max(new_depth, min_depth)

    return multiply_depth

def multi_resolution_feature_maps(feature_map_layout, depth_multiplier,
                                  min_depth, insert_1x1_conv, image_features,
                                  pool_residual=False):
    """Generates multi resolution feature maps from input image features.

    Generates multi-scale feature maps for detection as in the SSD papers by
    Liu et al: https://arxiv.org/pdf/1512.02325v2.pdf, See Sec 2.1.

    More specifically, it performs the following two tasks:
    1) If a layer name is provided in the configuration, returns that layer as a
       feature map.
    2) If a layer name is left as an empty string, constructs a new feature map
       based on the spatial shape and depth configuration. Note that the current
       implementation only supports generating new layers using convolution of
       stride 2 resulting in a spatial resolution reduction by a factor of 2.
       By default convolution kernel size is set to 3, and it can be customized
       by caller.

    An example of the configuration for Inception V3:
    {
      'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],
      'layer_depth': [-1, -1, -1, 512, 256, 128]
    }

    Args:
      feature_map_layout: Dictionary of specifications for the feature map
        layouts in the following format (Inception V2/V3 respectively):
        {
          'from_layer': ['Mixed_3c', 'Mixed_4c', 'Mixed_5c', '', '', ''],
          'layer_depth': [-1, -1, -1, 512, 256, 128]
        }
        or
        {
          'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],
          'layer_depth': [-1, -1, -1, 512, 256, 128]
        }
        If 'from_layer' is specified, the specified feature map is directly used
        as a box predictor layer, and the layer_depth is directly infered from the
        feature map (instead of using the provided 'layer_depth' parameter). In
        this case, our convention is to set 'layer_depth' to -1 for clarity.
        Otherwise, if 'from_layer' is an empty string, then the box predictor
        layer will be built from the previous layer using convolution operations.
        Note that the current implementation only supports generating new layers
        using convolutions of stride 2 (resulting in a spatial resolution
        reduction by a factor of 2), and will be extended to a more flexible
        design. Convolution kernel size is set to 3 by default, and can be
        customized by 'conv_kernel_size' parameter (similarily, 'conv_kernel_size'
        should be set to -1 if 'from_layer' is specified). The created convolution
        operation will be a normal 2D convolution by default, and a depthwise
        convolution followed by 1x1 convolution if 'use_depthwise' is set to True.
      depth_multiplier: Depth multiplier for convolutional layers.
      min_depth: Minimum depth for convolutional layers.
      insert_1x1_conv: A boolean indicating whether an additional 1x1 convolution
        should be inserted before shrinking the feature map.
      image_features: A dictionary of handles to activation tensors from the
        base feature extractor.
      pool_residual: Whether to add an average pooling layer followed by a
        residual connection between subsequent feature maps when the channel
        depth match. For example, with option 'layer_depth': [-1, 512, 256, 256],
        a pooling and residual layer is added between the third and forth feature
        map. This option is better used with Weight Shared Convolution Box
        Predictor when all feature maps have the same channel depth to encourage
        more consistent features across multi-scale feature maps.

    Returns:
      feature_maps: an OrderedDict mapping keys (feature map names) to
        tensors where each tensor has shape [batch, height_i, width_i, depth_i].

    Raises:
      ValueError: if the number entries in 'from_layer' and
        'layer_depth' do not match.
      ValueError: if the generated layer does not have the same resolution
        as specified.
    """
    depth_fn = get_depth_fn(depth_multiplier, min_depth)

    feature_map_keys = []
    feature_maps = []
    base_from_layer = ''
    use_explicit_padding = False
    if 'use_explicit_padding' in feature_map_layout:
        use_explicit_padding = feature_map_layout['use_explicit_padding']
    use_depthwise = False
    if 'use_depthwise' in feature_map_layout:
        use_depthwise = feature_map_layout['use_depthwise']
    for index, from_layer in enumerate(feature_map_layout['from_layer']):
        layer_depth = feature_map_layout['layer_depth'][index]
        conv_kernel_size = 3
        if 'conv_kernel_size' in feature_map_layout:
            conv_kernel_size = feature_map_layout['conv_kernel_size'][index]
        if from_layer:
            feature_map = image_features[from_layer]
            base_from_layer = from_layer
            feature_map_keys.append(from_layer)
        else:
            pre_layer = feature_maps[-1]
            pre_layer_depth = pre_layer.get_shape().as_list()[3]
            intermediate_layer = pre_layer
            if insert_1x1_conv:
                layer_name = '{}_1_Conv2d_{}_1x1_{}'.format(
                    base_from_layer, index, depth_fn(layer_depth / 2))
                intermediate_layer = slim.conv2d(
                    pre_layer,
                    depth_fn(layer_depth / 2), [1, 1],
                    padding='SAME',
                    stride=1,
                    scope=layer_name)
            layer_name = '{}_2_Conv2d_{}_{}x{}_s2_{}'.format(
                base_from_layer, index, conv_kernel_size, conv_kernel_size,
                depth_fn(layer_depth))
            stride = 2
            padding = 'SAME'
            if use_explicit_padding:
                padding = 'VALID'
                intermediate_layer = wmlt.fixed_padding(
                    intermediate_layer, conv_kernel_size)
            if use_depthwise:
                feature_map = slim.separable_conv2d(
                    intermediate_layer,
                    None, [conv_kernel_size, conv_kernel_size],
                    depth_multiplier=1,
                    padding=padding,
                    stride=stride,
                    scope=layer_name + '_depthwise')
                feature_map = slim.conv2d(
                    feature_map,
                    depth_fn(layer_depth), [1, 1],
                    padding='SAME',
                    stride=1,
                    scope=layer_name)
                if pool_residual and pre_layer_depth == depth_fn(layer_depth):
                    feature_map += slim.avg_pool2d(
                        pre_layer, [3, 3],
                        padding='SAME',
                        stride=2,
                        scope=layer_name + '_pool')
            else:
                feature_map = slim.conv2d(
                    intermediate_layer,
                    depth_fn(layer_depth), [conv_kernel_size, conv_kernel_size],
                    padding=padding,
                    stride=stride,
                    scope=layer_name)
            feature_map_keys.append(layer_name)
        feature_maps.append(feature_map)
    return collections.OrderedDict(
        [(x, y) for (x, y) in zip(feature_map_keys, feature_maps)])