import logging
import wmodule
from object_detection2.modeling.backbone.build import build_backbone
from object_detection2.modeling.proposal_generator.build import build_proposal_generator
from .build import META_ARCH_REGISTRY
import wsummary

@META_ARCH_REGISTRY.register()
class ProposalNetwork(wmodule.WModule):
    def __init__(self, cfg,parent=None,*args,**kwargs):
        del parent
        super().__init__(cfg,*args,**kwargs)
        self.backbone = build_backbone(cfg,parent=self)
        self.proposal_generator = build_proposal_generator(cfg,*args,**kwargs,parent=self)

    def forward(self, inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        features = self.backbone(inputs)
        outdata,proposal_losses = self.proposal_generator(inputs, features)
        #wsummary.detection_image_summary(images=inputs['image'],boxes=outdata['proposal_boxes'],name="proposal_boxes")
        return outdata,proposal_losses
