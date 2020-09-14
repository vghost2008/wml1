import os
import logging
import functools
import numpy as np
import tensorflow as tf
import wnnlayer as wnnl
from basic_tftools import channel as get_channel
from functools import partial
from collections import OrderedDict
import wml_tfutils as wmlt

slim = tf.contrib.slim


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class BasicBlock(object):
    expansion = 1

    def __init__(self, num_outputs, stride=1, downsample=None,activation_fn=None,normalizer_fn=None,normalizer_params=None):
        assert normalizer_fn is not None
        self.num_outputs = num_outputs
        self.normalizer_fn = normalizer_fn
        if normalizer_params is None:
            self.normalizer_params = {}
        else:
            self.normalizer_params = normalizer_params
        self.activation_fn = activation_fn
        self.stride = stride
        self.downsample = downsample

    def __call__(self, x,scope=None):
        with tf.variable_scope(scope,default_name="BasicBlock"):
            residual = x
            out = slim.conv2d(x,self.num_outputs,3,self.stride,activation_fn=self.activation_fn,
                              normalizer_fn=self.normalizer_fn,
                              normalizer_params=self.normalizer_params)
            out = slim.conv2d(out,self.num_outputs,3,1,
                             normalizer_fn=self.normalizer_fn,
                             normalizer_params=self.normalizer_params)

            if self.downsample is not None:
                residual = self.downsample(x)
            if get_channel(residual) != get_channel(out):
                residual = slim.conv2d(out,self.num_outputs,1,1,
                                       activation_fn=None,
                                       normalizer_fn=None)

            out += residual
            if self.activation_fn is not None:
                out = self.activation_fn(out)
            return out


class Bottleneck(object):
    expansion = 4
    def __init__(self, num_outputs, stride=1, downsample=None,activation_fn=None,normalizer_fn=None,normalizer_params=None):

        assert normalizer_fn is not None
        self.num_outputs = num_outputs
        self.normalizer_fn = normalizer_fn
        if normalizer_params is None:
            self.normalizer_params = {}
        else:
            self.normalizer_params = normalizer_params
        self.activation_fn = activation_fn
        self.stride = stride
        self.downsample = downsample

    def __call__(self, x,scope=None):
        with tf.variable_scope(scope,default_name="Bottleneck"):
            residual = x
            out = slim.conv2d(x,self.num_outputs,3,1,activation_fn=self.activation_fn,
                              normalizer_fn=self.normalizer_fn,
                              normalizer_params=self.normalizer_params)
            out = slim.conv2d(out,self.num_outputs,3,self.stride,activation_fn=self.activation_fn,
                              normalizer_fn=self.normalizer_fn,
                              normalizer_params=self.normalizer_params)
            out = slim.conv2d(out,self.num_outputs*self.expansion,3,1,
                             activation_fn=None,
                             normalizer_fn=self.normalizer_fn,
                             normalizer_params=self.normalizer_params)

            if self.downsample is not None:
                residual = self.downsample(x)
            if get_channel(residual) != get_channel(out):
                residual = slim.conv2d(out,self.num_outputs*self.expansion,1,1,
                             activation_fn=None,
                             normalizer_fn=None)

            out += residual
            if self.activation_fn is not None:
                out = self.activation_fn(out)
            return out

class HighResolutionModule(object):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True,fuse=True,
                 activation_fn=None, normalizer_fn=None, normalizer_params=None):
        assert normalizer_fn is not None
        self.fuse = fuse
        self.normalizer_fn = normalizer_fn
        if normalizer_params is None:
            self.normalizer_params = {}
        else:
            self.normalizer_params = normalizer_params
        self.activation_fn = activation_fn

        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_channels = num_channels
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.blocks = blocks
        self.num_blocks = num_blocks

        self.multi_scale_output = multi_scale_output

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def downsample(self,x,expansion,stride,scope=None):
        with tf.variable_scope(scope,default_name="downsample"):
            num_outputs = get_channel(x)*expansion
            out = slim.conv2d(x, num_outputs, 3, stride,
                          normalizer_fn=self.normalizer_fn,
                          normalizer_params=self.normalizer_params)
            return out

    def _make_one_branch(self, x,branch_index, block, num_blocks, num_channels,
                         stride=1,scope=None):
        with tf.variable_scope(scope,f"one_branch_{branch_index}"):
            downsample = None
            if stride != 1 or \
                    self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
                downsample = self.downsample

            x = block(num_channels[branch_index], stride, downsample,
                      activation_fn=self.activation_fn,
                      normalizer_fn=self.normalizer_fn,
                      normalizer_params=self.normalizer_params)(x,scope="block0")
            self.num_inchannels[branch_index] = \
                num_channels[branch_index] * block.expansion
            for i in range(1, num_blocks[branch_index]):
                x = block(num_channels[branch_index],
                          activation_fn=self.activation_fn,
                          normalizer_fn=self.normalizer_fn,
                          normalizer_params=self.normalizer_params)(x,scope=f"block{i}")
            return x

    def fuse_layers(self,i,j,x,scope=None):
        if self.num_branches == 1:
            return None
        with tf.variable_scope(scope,default_name=f"fuse_layer{i}_{j}"):
            num_inchannels = self.num_inchannels
            if j > i:
                x = slim.conv2d(x, num_inchannels[i], 1, 1, activation_fn=None,
                                  normalizer_fn=self.normalizer_fn,
                                  normalizer_params=self.normalizer_params)
                x = wnnl.upsample(x,scale_factor=2 ** (j - i), mode=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            elif j == i:
                return x
            else:
                for k in range(i - j):
                    if k == i - j - 1:
                        num_outchannels_conv3x3 = num_inchannels[i]
                        x = slim.conv2d(x, num_outchannels_conv3x3,
                                          3, 2, activation_fn=None,
                                          normalizer_fn=self.normalizer_fn,
                                          normalizer_params=self.normalizer_params,
                                          padding="SAME")
                    else:
                        num_outchannels_conv3x3 = num_inchannels[j]
                        x = slim.conv2d(x, num_outchannels_conv3x3,
                                        3, 2, activation_fn=self.activation_fn,
                                        normalizer_fn=self.normalizer_fn,
                                        normalizer_params=self.normalizer_params,
                                        padding="SAME")

            return x

    def get_num_inchannels(self):
        return self.num_inchannels

    def __call__(self, x,scope=None):
        with tf.variable_scope(scope,default_name="HighResolutionModule"):
            if self.num_branches == 1:
                return [self._make_one_branch(x[0],0, self.blocks, self.num_blocks, self.num_channels,scope="branch")]

            for i in range(self.num_branches):
                x[i] = self._make_one_branch(x[i],i, self.blocks, self.num_blocks, self.num_channels,scope=f"branch{i}")

            if not self.fuse:
                return x

            return self.fuse_layer(x)

    def fuse_layer(self, xs):
        with tf.variable_scope("Fuse"):
            ys = []
            for i,v0 in enumerate(xs):
                shape0 = wmlt.combined_static_and_dynamic_shape(v0)
                datas = []
                for j,v1 in enumerate(xs):
                    if i!=j:
                        chl = get_channel(v0)
                        v1 = tf.image.resize_nearest_neighbor(v1, shape0[1:3])
                        v1 = slim.conv2d(v1, chl, [3, 3],
                                         activation_fn=None,
                                         normalizer_fn=self.normalizer_fn,
                                         normalizer_params=self.normalizer_params,
                                         scope=f"smooth{i}_{j}")
                    datas.append(v1)
                v = tf.add_n(datas)/len(datas)
                ys.append(v)

            return ys



blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}
cfg_w32 = {
    'STAGE1':{'NUM_CHANNELS':[64],'BLOCK':'BOTTLENECK','NUM_BLOCKS':[4],'NUM_MODULES':1},
    'STAGE2': {'NUM_CHANNELS': [32,64], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4,4],'NUM_BRANCHES':2,'NUM_MODULES':1},
    'STAGE3': {'NUM_CHANNELS': [32,64,128], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4,4,4], 'NUM_BRANCHES':3,'NUM_MODULES':4},
    'STAGE4': {'NUM_CHANNELS': [32,64,128,256], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4,4,4,4], 'NUM_BRANCHES':4,'NUM_MODULES':3},

}
mini_cfg_w32 = {
    'STAGE1':{'NUM_CHANNELS':[32],'BLOCK':'BOTTLENECK','NUM_BLOCKS':[3],'NUM_MODULES':1},
    'STAGE2': {'NUM_CHANNELS': [32,64], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [3,3],'NUM_BRANCHES':2,'NUM_MODULES':1},
    'STAGE3': {'NUM_CHANNELS': [32,64,128], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [3,3,3], 'NUM_BRANCHES':3,'NUM_MODULES':2},
    'STAGE4': {'NUM_CHANNELS': [32,64,128,256], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [3,3,3,3], 'NUM_BRANCHES':4,'NUM_MODULES':2},
}

class HighResolutionNet(object):

    def __init__(self, cfg=cfg_w32,output_channel=None, **kwargs):
        self.cfg = cfg
        self.stage1_cfg = cfg['STAGE1']
        self.stage2_cfg = cfg['STAGE2']
        self.stage3_cfg = cfg['STAGE3']
        self.stage4_cfg = cfg['STAGE4']
        self.activation_fn = None
        self.normalizer_fn = wnnl.evo_norm_s0
        self.normalizer_params = None
        self.output_channel = output_channel

    def _make_transition_layer(self,xs,
            num_channels_pre_layer, num_channels_cur_layer,scope=None):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        ys = []
        with tf.variable_scope(scope,default_name="transition_layer"):
            out_shapes = []
            for i in range(num_branches_cur):
                if i < num_branches_pre:
                    out_shapes.append(wmlt.combined_static_and_dynamic_shape(xs[i])[1:3])
                else:
                    last_shape = out_shapes[-1]
                    h = last_shape[0]//2
                    w = last_shape[1]//2
                    out_shapes.append([h,w])

            with tf.variable_scope("Fuse"):
                for i in range(num_branches_cur):
                    shape0 = out_shapes[i]
                    datas = []
                    for j, v1 in enumerate(xs):
                        if i != j:
                            chl = num_channels_cur_layer[i]
                            v1 = tf.image.resize_nearest_neighbor(v1, shape0)
                            v1 = slim.conv2d(v1, chl, [3, 3],
                                             activation_fn=None,
                                             normalizer_fn=self.normalizer_fn,
                                             normalizer_params=self.normalizer_params,
                                             scope=f"smooth{i}_{j}")
                        datas.append(v1)
                    if len(datas)>1:
                        v = tf.add_n(datas) / len(datas)
                    else:
                        v = datas[0]
                    ys.append(v)

        return ys


    def downsample(self,x,expansion,stride,scope=None):
        with tf.variable_scope(scope,default_name="downsample"):
            num_outputs = get_channel(x)*expansion
            out = slim.conv2d(x, num_outputs, 3, stride,
                              normalizer_fn=self.normalizer_fn,
                              normalizer_params=self.normalizer_params)
            return out

    def _make_layer(self, x,block, planes, blocks, stride=1,scope=None):
        with tf.variable_scope(scope,default_name="layer"):
            downsample = None
            inplanes = get_channel(x)
            if stride != 1 or inplanes != planes * block.expansion:
                downsample = partial(self.downsample,expansion=block.expansion,stride=stride)

            x = block(planes, stride, downsample,normalizer_fn=self.normalizer_fn,
                      normalizer_params=self.normalizer_params,
                      activation_fn=self.activation_fn)(x,scope="block0")
            for i in range(1, blocks):
                x = block(planes,normalizer_fn=self.normalizer_fn,
                          normalizer_params=self.normalizer_params,
                          activation_fn=self.activation_fn)(x,scope=f"block{i}")

        return x


    def _make_stage(self, xs,layer_config, num_inchannels,
                    multi_scale_output=True,fuse=False,scope=None):
        with tf.variable_scope(scope,default_name="scope"):
            num_modules = layer_config['NUM_MODULES']
            num_blocks = layer_config['NUM_BLOCKS']
            num_channels = layer_config['NUM_CHANNELS']
            num_branches = len(num_channels)
            block = blocks_dict[layer_config['BLOCK']]
            fuse_method = layer_config.get('FUSE_METHOD','SUM')

            for i in range(num_modules):
                if not multi_scale_output and i == num_modules - 1:
                    reset_multi_scale_output = False
                else:
                    reset_multi_scale_output = True

                module = HighResolutionModule(num_branches,
                                                block,
                                                num_blocks,
                                                num_inchannels,
                                                num_channels,
                                                fuse_method,
                                                reset_multi_scale_output,
                                              fuse=True if fuse else (i!=num_modules-1),
                                              activation_fn=self.activation_fn,
                                              normalizer_fn=self.normalizer_fn,
                                              normalizer_params=self.normalizer_params)
                xs = module(xs,scope=f"HighResolutionModule{i}")
                num_inchannels = module.get_num_inchannels()

            return xs, num_inchannels


    def __call__(self, x):
        num_channels = 64
        with tf.variable_scope("HRNet"):
            x = slim.conv2d(x, num_channels,
                            3, 2, activation_fn=self.activation_fn,
                            normalizer_fn=self.normalizer_fn,
                            normalizer_params=self.normalizer_params,
                            padding="SAME")
            x = slim.conv2d(x, num_channels,
                            3, 2, activation_fn=self.activation_fn,
                            normalizer_fn=self.normalizer_fn,
                            normalizer_params=self.normalizer_params,
                            padding="SAME")
            num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
            block = blocks_dict[self.stage1_cfg['BLOCK']]
            num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]

            x = self._make_layer(x,block, num_channels, num_blocks,scope="stage1")

            num_channels = self.stage2_cfg['NUM_CHANNELS']
            block = blocks_dict[self.stage2_cfg['BLOCK']]
            num_channels = [
                num_channels[i] * block.expansion for i in range(len(num_channels))]
            stage1_out_channel = block.expansion * num_channels
            x_list = self._make_transition_layer([x],[stage1_out_channel], num_channels)
            x_list,pre_stage_channels = self._make_stage(x_list,self.stage2_cfg, num_channels,scope="stage2")

            num_channels = self.stage3_cfg['NUM_CHANNELS']
            block = blocks_dict[self.stage3_cfg['BLOCK']]
            num_channels = [
                num_channels[i] * block.expansion for i in range(len(num_channels))]
            x_list = self._make_transition_layer(x_list, pre_stage_channels, num_channels)
            x_list,pre_stage_channels = self._make_stage(x_list,self.stage3_cfg, num_channels,scope="stage3")

            num_channels = self.stage4_cfg['NUM_CHANNELS']
            block = blocks_dict[self.stage4_cfg['BLOCK']]
            num_channels = [
                num_channels[i] * block.expansion for i in range(len(num_channels))]
            x_list = self._make_transition_layer(x_list, pre_stage_channels, num_channels)
            x_list,pre_stage_channels = self._make_stage(x_list,self.stage4_cfg, num_channels,scope="stage4",
                                                         fuse=True)

            endpoints = OrderedDict()
            with tf.variable_scope('output'):
                for i,x in enumerate(x_list):
                    if self.output_channel is not None:
                        x = slim.conv2d(x,self.output_channel,1,1,activation_fn=None,normalizer_fn=None,
                                        scope=f"output_smooth_{i}")
                    endpoints[f"C{i+2}"] = x
            return endpoints

