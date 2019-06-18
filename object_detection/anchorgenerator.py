import numpy as np
from wtfop.wtfop_ops import boxes_encode,multi_anchor_generator,anchor_generator
import wml_utils as wmlu
import tensorflow as tf
import math

class MultiscaleGridAnchorGenerator(object):
    '''
    aspect_ratios:like [1.0,2.0,0.5]
    '''
    def __init__(self, min_level, max_level, aspect_ratios,anchor_scale=4.0,
               scales_per_octave=2):
        aspect_ratios = [1.0/x for x in aspect_ratios]
        aspect_ratios.reverse()
        self._aspect_ratios = aspect_ratios
        self._scales_per_octave = scales_per_octave
        self._min_level = min_level
        self._max_level = max_level
        self.anchor_scale = anchor_scale
        self.scales_per_level = [2**(float(scale) / scales_per_octave)
                  for scale in range(scales_per_octave)]

    '''
    feature_map_shape_list: like [[20,20],[30,30]]
    size:输入图像的大小
    '''
    def __call__(self, feature_map_shape_list, size=[640.0,640.0]):
        raw_scales = []
        anchors = []
        for i,l in enumerate(range(self._min_level,self._max_level+1)):
            level_scales = []
            for s in self.scales_per_level:
                level_scales.append(s*self.anchor_scale*math.pow(2,l))
            raw_scales.append(level_scales)
            anchors.append(anchor_generator(shape=feature_map_shape_list[i],size=size,scales=level_scales,aspect_ratios=self._aspect_ratios))
        res = tf.concat(anchors,axis=0)
        return tf.expand_dims(res,axis=0)


class SSDGridAnchorGenerator(object):
    def __init__(self,min_scale=0.2,max_scale=0.95,
                    aspect_ratios=(1.0, 2.0, 3.0, 1.0 / 2, 1.0 / 3),
                    scales=None,
                    interpolated_scale_aspect_ratio=1.0,
                    reduce_boxes_in_lowest_layer=True):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.interpolated_scale_aspect_ratio = interpolated_scale_aspect_ratio
        self.reduce_boxes_in_lowest_layer = reduce_boxes_in_lowest_layer
        self.box_specs_list = None

    def __call__(self,feature_map_shape_list,
                       size=[1.0,1.0]):
        num_layers = len(feature_map_shape_list)
        box_specs_list = []

        if self.scales is None or not self.scales:
            scales = [self.min_scale + (self.max_scale - self.min_scale) * i / (num_layers - 1)
                      for i in range(num_layers)] + [1.0]
        else:
            # Add 1.0 to the end, which will only be used in scale_next below and used
            # for computing an interpolated scale for the largest scale in the list.
            scales = self.scales+[1.0]

        tf_anchors=[]
        for layer, scale, scale_next,shape in zip(
                range(num_layers), scales[:-1], scales[1:],self.feature_maps_shape):
            layer_box_specs = []
            if layer == 0 and self.reduce_boxes_in_lowest_layer:
                layer_box_specs = [(0.1, 1.0), (scale, 2.0), (scale, 0.5)]
            else:
                for aspect_ratio in self.aspect_ratios:
                    layer_box_specs.append((scale, aspect_ratio))
                # Add one more anchor, with a scale between the current scale, and the
                # scale for the next layer, with a specified aspect ratio (1.0 by
                # default).
                if self.interpolated_scale_aspect_ratio > 0.0:
                    layer_box_specs.append((np.sqrt(scale * scale_next),
                                            self.interpolated_scale_aspect_ratio))
            box_specs_list.append(layer_box_specs)

            tf_anchors.append(SSDGridAnchorGenerator.get_a_layer_anchors(layer_box_specs=layer_box_specs,shape=shape,size=size))
        wmlu.show_list(box_specs_list)
        self.box_specs_list = box_specs_list
        anchors = tf.concat(tf_anchors,axis=0)
        anchors = tf.expand_dims(anchors,axis=0)
        return anchors

    @staticmethod
    def get_a_layer_anchors(layer_box_specs,shape,size):
        scales,ratios = zip(*layer_box_specs)
        return multi_anchor_generator(shape=shape,size=size,scales=scales,aspect_ratios=ratios)




