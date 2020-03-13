#coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
import object_detection2.od_toolkit as odt

slim = tf.contrib.slim

'''
feature_maps:shape=[[h,w,c0],[2*h,2*w,c1],[4*h,4*w,c2],...]
'''
def BiFPN(feature_maps,conv_op=slim.conv2d,upsample_op=wmlt.nearest_neighbor_upsampling,depths=None,scope=None):
    mid_feature_maps = []
    out_feature_maps = []
    if depths is None:
        depths = feature_maps[0].get_shape().as_list()[-1]
    with tf.variable_scope(scope,"fpn_top_down_bottom_up"):
        last = tf.identity(feature_maps[0])
        mid_feature_maps.append(last)
        for i in range(1,len(feature_maps)):
            last = upsample_op(last,scale=2,scope=f"upsample{i}")
            net = odt.fusion([last,feature_maps[i]],depth=depths,scope=f"td_fusion{i}")
            net = conv_op(net,net.get_shape().as_list()[-1],kernel_size=3,scope=f"td_smooth{i}")
            last = net
            mid_feature_maps.append(net)

        last = tf.identity(mid_feature_maps[-1])
        out_feature_maps.append(last)
        for i in reversed(range(len(mid_feature_maps)-1)):
            net = mid_feature_maps[i]
            last = slim.max_pool2d(last, [2, 2], padding='SAME', stride=2, scope=f"max_pool{i}")
            if i>0:
                net = odt.fusion([feature_maps[i],last,net],depth=depths,scope=f"bu_fusion{i}")
            else:
                net = odt.fusion([last,net],depth=depths,scope=f"bu_fusion{i}")
            net = conv_op(net,net.get_shape().as_list()[-1],kernel_size=3,scope=f"bu_smooth{i}")
            last = net
            out_feature_maps.append(net)
        out_feature_maps.reverse()

    return out_feature_maps