#coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
import object_detection2.od_toolkit as odt

slim = tf.contrib.slim

'''
EfficientDet: Scalable and Efficient Object Detection
官方实现中都是op_after_combine都是BN(conv(swish(x)))
feature_maps:shape=[[h,w,c0],[2*h,2*w,c1],[4*h,4*w,c2],...]
'''
def BiFPN(feature_maps,conv_op=slim.conv2d,upsample_op=wmlt.nearest_neighbor_upsampling,activation_fn=tf.nn.swish,
          depths=None,scope=None):
    mid_feature_maps = []
    out_feature_maps = []
    if depths is None:
        depths = feature_maps[0].get_shape().as_list()[-1]
    with tf.variable_scope(scope,"fpn_top_down_bottom_up"):
        last = feature_maps[0]
        mid_feature_maps.append(last)
        for i in range(1,len(feature_maps)):
            with tf.variable_scope(f"down_node{i}"):
                last = upsample_op(last,scale=2,scope=f"upsample{i}")
                net = odt.fusion([last,feature_maps[i]],depth=depths,scope=f"td_fusion{i}")
                if activation_fn is not None:
                    with tf.name_scope(f"op_after_combine{i}"):
                        net = activation_fn(net)
                net = conv_op(net,net.get_shape().as_list()[-1],kernel_size=3,
                              activation_fn=None,
                              scope=f"op_after_combine{i}")
                last = net
                mid_feature_maps.append(net)

        last = mid_feature_maps[-1]
        out_feature_maps.append(last)
        for i in reversed(range(len(mid_feature_maps)-1)):
            with tf.variable_scope(f"up_node{i}"):
                net = mid_feature_maps[i]
                last = slim.max_pool2d(last, [2, 2], padding='SAME', stride=2, scope=f"max_pool{i}")
                if i>0:
                    net = odt.fusion([feature_maps[i],last,net],depth=depths,scope=f"bu_fusion{i}")
                else:
                    net = odt.fusion([last,net],depth=depths,scope=f"bu_fusion{i}")
                if activation_fn is not None:
                    with tf.name_scope(f"op_after_combine{i}"):
                        net = activation_fn(net)
                net = conv_op(net,net.get_shape().as_list()[-1],kernel_size=3,
                              scope=f"op_after_combine{i}",
                              activation_fn=None)
                last = net
                out_feature_maps.append(net)
        out_feature_maps.reverse()

    return out_feature_maps