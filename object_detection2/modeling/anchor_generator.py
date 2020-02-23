#coding=utf-8
from thirdparty.registry import Registry
import wmodule
import wtfop.wtfop_ops as wop
import wml_tfutils as wmlt
import tensorflow as tf
import wsummary
from object_detection2.datadef import *

ANCHOR_GENERATOR_REGISTRY = Registry("ANCHOR_GENERATOR")

def build_anchor_generator(cfg, *args,**kwargs):
    """
    Built an anchor generator from `cfg.MODEL.ANCHOR_GENERATOR.NAME`.
    """
    anchor_generator = cfg.MODEL.ANCHOR_GENERATOR.NAME
    return ANCHOR_GENERATOR_REGISTRY.get(anchor_generator)(*args,cfg=cfg,**kwargs)

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
        #aspect_ratios = [[1/2,1:1,2/1],...]
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
        with tf.name_scope("show_anchors"):
            image = tf.ones(img_size)
            image = tf.expand_dims(image,axis=0)
            image = tf.expand_dims(image,axis=-1)
            image = tf.tile(image,[1,1,1,3])
            for i in range(len(anchors)):
                num_cell_anchors = len(self.aspect_ratios[i])*len(self.sizes[i])
                shape = wmlt.combined_static_and_dynamic_shape(features[i])
                offset = ((shape[1]//2)*shape[2]+shape[2]//2)*num_cell_anchors
                boxes = anchors[i][offset:offset+num_cell_anchors]
                boxes = tf.expand_dims(boxes,axis=0)
                wsummary.detection_image_summary(images=image,boxes=boxes,name=f"level_{i}")


    @property
    def num_cell_anchors(self):
        return [len(x[0])*len(x[1]) for x in zip(self.sizes,self.aspect_ratios)]

    def forward(self, inputs,features,):
        anchors = []
        image = inputs['image']
        with tf.name_scope("anchor_generator"):
            size = wmlt.combined_static_and_dynamic_shape(image)[1:3]
            for i,feature in enumerate(features):
                shape = wmlt.combined_static_and_dynamic_shape(feature)
                anchors.append(wop.anchor_generator(shape=shape[1:3],size=size,
                                                    scales=self.sizes[i],
                                                    aspect_ratios=self.aspect_ratios[i]))
        if self.cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
            self.show_anchors(anchors,features,img_size=size)
        return anchors
