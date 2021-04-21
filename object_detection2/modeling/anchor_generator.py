#coding=utf-8
from thirdparty.registry import Registry
import wmodule
import wml_tfutils as wmlt
import tensorflow as tf
import wsummary
from object_detection2.datadef import *
from collections import Iterable
import math
import numpy as np

ANCHOR_GENERATOR_REGISTRY = Registry("ANCHOR_GENERATOR")

def build_anchor_generator(cfg, *args,**kwargs):
    """
    Built an anchor generator from `cfg.MODEL.ANCHOR_GENERATOR.NAME`.
    """
    anchor_generator = cfg.MODEL.ANCHOR_GENERATOR.NAME
    return ANCHOR_GENERATOR_REGISTRY.get(anchor_generator)(*args,cfg=cfg,**kwargs)

class AnchorGeneratorF(object):
    def __init__(self,scales,aspect_ratios):
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = self.get_cell_anchors()

    def get_cell_anchors(self):
        anchors = []
        for a in self.aspect_ratios:
            for s in self.scales:
                sa = math.sqrt(a)
                hw = s*sa/2
                hh = s/(sa*2)
                box = [-hh,-hw,hh,hw]
                anchors.append(box)

        return np.array(anchors,dtype=np.float32)


    def __call__(self,shape,size):
        with tf.name_scope("AnchorGeneratorF"):
            y = tf.range(shape[0])
            x = tf.range(shape[1])
            X,Y = tf.meshgrid(x,y)
            offset = 0.5
            if not isinstance(shape[0],(int,float)):
                shapef = tf.to_float(shape)
            else:
                shapef = shape

            if not isinstance(size[0],(int,float)):
                sizef = tf.to_float(size)
            else:
                sizef = size
            pos_scale = tf.convert_to_tensor([1.0/shapef[0],1.0/shapef[1],1.0/shapef[0],1.0/shapef[1]],
                                         dtype=tf.float32)
            box_scale = tf.convert_to_tensor([1.0/sizef[0],1.0/sizef[1],1.0/sizef[0],1.0/sizef[1]],
                                             dtype=tf.float32)
            pos_scale = tf.reshape(pos_scale,[1,1,4])
            box_scale = tf.reshape(box_scale,[1,4])
            data = tf.cast(tf.stack([Y,X,Y,X],axis=-1),tf.float32)+tf.convert_to_tensor(offset)
            data = data*pos_scale
            data = tf.reshape(data,[shape[0],shape[1],1,4])
            cell_anchors = self.cell_anchors*box_scale
            cell_anchors = tf.reshape(cell_anchors,[1,1,-1,4])
            anchors = data+cell_anchors
            anchors = tf.reshape(anchors,[-1,4])
            return anchors

def anchor_generator(shape,size,scales,aspect_ratios):
    ag = AnchorGeneratorF(scales=scales,aspect_ratios=aspect_ratios)
    return ag(shape=shape,size=size)

@ANCHOR_GENERATOR_REGISTRY.register()
class DefaultAnchorGenerator(wmodule.WChildModule):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    """

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        cfg = self.cfg
        # fmt: off
        #sizes = [[128,256],[256,384]]
        self.sizes         = cfg.MODEL.ANCHOR_GENERATOR.SIZES
        '''
        aspect_ratios = [[1/2,1:1,2/1],...]
        or:
        aspect_ratios = [[[1/2,1:1,2/1],[1/3,1,3]],...]
        w/h
        '''
        self.aspect_ratios = cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS
        if len(self.aspect_ratios) == 1 and len(self.sizes)>1:
            self.aspect_ratios = self.aspect_ratios*len(self.sizes)


    @property
    def box_dim(self):
        """
        Returns:
            int: the dimension of each anchor box.
        """
        return 4

    def show_anchors(self,anchors,features,img_size=[512,512]):
        with tf.device(":/cpu:0"):
            with tf.name_scope("show_anchors"):
                image = tf.ones(img_size)
                image = tf.expand_dims(image,axis=0)
                image = tf.expand_dims(image,axis=-1)
                image = tf.tile(image,[1,1,1,3])
                for i in range(len(anchors)):
                    if not isinstance(self.aspect_ratios[i][0],Iterable):
                        num_cell_anchors = len(self.aspect_ratios[i])*len(self.sizes[i])
                    else:
                        num_cell_anchors = len(self.aspect_ratios[i][0]) * len(self.sizes[i])
                    shape = wmlt.combined_static_and_dynamic_shape(features[i])
                    offset = ((shape[1]//2)*shape[2]+shape[2]//2)*num_cell_anchors
                    boxes = anchors[i][offset:offset+num_cell_anchors]
                    boxes = tf.expand_dims(boxes,axis=0)
                    wsummary.detection_image_summary(images=image,boxes=boxes,name=f"level_{i}")


    @property
    def num_cell_anchors(self):
        if not isinstance(self.aspect_ratios[0][0], Iterable):
            return [len(x[0])*len(x[1]) for x in zip(self.sizes,self.aspect_ratios)]
        else:
            return [len(x[0])*len(x[1][0]) for x in zip(self.sizes,self.aspect_ratios)]

    def forward(self, inputs,features):
        anchors = []
        image = inputs['image']
        assert len(features)==len(self.sizes),"Error features len."
        with tf.name_scope("anchor_generator"):
            size = wmlt.combined_static_and_dynamic_shape(image)[1:3]
            for i,feature in enumerate(features):
                shape = wmlt.combined_static_and_dynamic_shape(feature)
                '''
                anchor_generator反回的为[N,4]
                表示每个位置，每个比率，每个尺寸的anchor box
                即[box0(s0,r0),box1(s1,r0),box2(s0,r1),box3(s1,r1),...]
                '''
                if not isinstance(self.aspect_ratios[i][0],Iterable):
                    anchors.append(anchor_generator(shape=shape[1:3],size=size,
                                                        scales=self.sizes[i],
                                                        aspect_ratios=self.aspect_ratios[i]))
                else:
                    assert len(self.sizes[i])==len(self.aspect_ratios[i]),"error size"
                    tanchors = []
                    for j in range(len(self.sizes[i])):
                        tanchors.append(anchor_generator(shape=shape[1:3], size=size,
                                                            scales=[self.sizes[i][j]],
                                                            aspect_ratios=self.aspect_ratios[i][j]))
                    tanchors = [tf.expand_dims(x,axis=1) for x in tanchors]
                    tanchors = tf.concat(tanchors,axis=1)
                    tanchors = tf.reshape(tanchors,[-1,4])
                    anchors.append(tanchors)
        if self.cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
            self.show_anchors(anchors,features,img_size=size)
        return anchors


@ANCHOR_GENERATOR_REGISTRY.register()
class AnchorGenerator(wmodule.WChildModule):
    """
    为每一个size,anchor对生成一个anchor
    """

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        cfg = self.cfg
        # fmt: off
        #sizes = [[128,256],[256,384]]
        self.sizes = []
        '''
        aspect_ratios = [[1/2,1:1,2/1],...]
        or:
        aspect_ratios = [[[1/2,1:1,2/1],[1/3,1,3]],...]
        w/h
        '''
        self.aspect_ratios = []
        if len(self.aspect_ratios) == 1 and len(self.sizes)>1:
            self.aspect_ratios = self.aspect_ratios*len(self.sizes)
        for i in range(len(cfg.MODEL.ANCHOR_GENERATOR.SIZES)):
            for j,s in enumerate(cfg.MODEL.ANCHOR_GENERATOR.SIZES[i]):
                for k,r in enumerate(cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS[i][j]):
                    self.sizes.append([s])
                    self.aspect_ratios.append([r])


    @property
    def box_dim(self):
        """
        Returns:
            int: the dimension of each anchor box.
        """
        return 4

    def show_anchors(self,anchors,features,img_size=[512,512]):
        with tf.device(":/cpu:0"):
            with tf.name_scope("show_anchors"):
                image = tf.ones(img_size)
                image = tf.expand_dims(image,axis=0)
                image = tf.expand_dims(image,axis=-1)
                image = tf.tile(image,[1,1,1,3])
                for i in range(len(anchors)):
                    if not isinstance(self.aspect_ratios[i][0],Iterable):
                        num_cell_anchors = len(self.aspect_ratios[i])*len(self.sizes[i])
                    else:
                        num_cell_anchors = len(self.aspect_ratios[i][0]) * len(self.sizes[i])
                    shape = wmlt.combined_static_and_dynamic_shape(features[i])
                    offset = ((shape[1]//2)*shape[2]+shape[2]//2)*num_cell_anchors
                    boxes = anchors[i][offset:offset+num_cell_anchors]
                    boxes = tf.expand_dims(boxes,axis=0)
                    wsummary.detection_image_summary(images=image,boxes=boxes,name=f"level_{i}")


    @property
    def num_cell_anchors(self):
        if not isinstance(self.aspect_ratios[0][0], Iterable):
            return [len(x[0])*len(x[1]) for x in zip(self.sizes,self.aspect_ratios)]
        else:
            return [len(x[0])*len(x[1][0]) for x in zip(self.sizes,self.aspect_ratios)]

    def forward(self, inputs,features):
        anchors = []
        image = inputs['image']
        assert len(features)==len(self.sizes),f"Error features len {len(features)} vs {len(self.sizes)}."
        with tf.name_scope("anchor_generator"):
            size = wmlt.combined_static_and_dynamic_shape(image)[1:3]
            for i,feature in enumerate(features):
                shape = wmlt.combined_static_and_dynamic_shape(feature)
                '''
                anchor_generator反回的为[N,4]
                表示每个位置，每个比率，每个尺寸的anchor box
                即[box0(s0,r0),box1(s1,r0),box2(s0,r1),box3(s1,r1),...]
                '''
                if not isinstance(self.aspect_ratios[i][0],Iterable):
                    anchors.append(anchor_generator(shape=shape[1:3],size=size,
                                                        scales=self.sizes[i],
                                                        aspect_ratios=self.aspect_ratios[i]))
                else:
                    assert len(self.sizes[i])==len(self.aspect_ratios[i]),"error size"
                    tanchors = []
                    for j in range(len(self.sizes[i])):
                        tanchors.append(anchor_generator(shape=shape[1:3], size=size,
                                                            scales=[self.sizes[i][j]],
                                                            aspect_ratios=self.aspect_ratios[i][j]))
                    tanchors = [tf.expand_dims(x,axis=1) for x in tanchors]
                    tanchors = tf.concat(tanchors,axis=1)
                    tanchors = tf.reshape(tanchors,[-1,4])
                    anchors.append(tanchors)
        if self.cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
            self.show_anchors(anchors,features,img_size=size)
        return anchors