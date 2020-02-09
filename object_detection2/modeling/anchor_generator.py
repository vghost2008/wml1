#coding=utf-8
from thirdparty.registry import Registry
import wmodule
import wtfop.wtfop_ops as wop
import wml_tfutils as wmlt

ANCHOR_GENERATOR_REGISTRY = Registry("ANCHOR_GENERATOR")

def build_anchor_generator(cfg, *args,**kwargs):
    """
    Built an anchor generator from `cfg.MODEL.ANCHOR_GENERATOR.NAME`.
    """
    anchor_generator = cfg.MODEL.ANCHOR_GENERATOR.NAME
    return ANCHOR_GENERATOR_REGISTRY.get(anchor_generator)(*args,**kwargs)

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

    @property
    def box_dim(self):
        """
        Returns:
            int: the dimension of each anchor box.
        """
        return 4

    @property
    def num_cell_anchors(self):
        return [len(x[0])*len(x[1]) for x in zip(self.sizes,self.aspect_ratios)]

    def forward(self, inputs,features,):
        anchors = []
        image = inputs['image']
        size = wmlt.combined_static_and_dynamic_shape(image)[1:3]
        for i,feature in enumerate(features):
            shape = wmlt.combined_static_and_dynamic_shape(feature)
            anchors.append(wop.anchor_generator(shape=shape[1:3],size=size,
                                                scales=self.sizes[i],
                                                aspect_ratios=self.aspect_ratios[i]))
        return anchors