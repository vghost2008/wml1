#coding=utf-8
import tensorflow as tf
from .build import META_ARCH_REGISTRY
from object_detection2.modeling.build import build_outputs
from object_detection2.modeling.backbone.build import build_backbone
from object_detection2.standard_names import *
from .meta_arch import MetaArch
from object_detection2.datadef import *
from object_detection2.modeling.semantic_heads.build import build_semantic_head
import wsummary
import numpy as np

slim = tf.contrib.slim

__all__ = ["DeepLab"]

@META_ARCH_REGISTRY.register()
class DeepLab(MetaArch):

    def __init__(self, cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)

        # fmt: off
        self.num_classes              = cfg.MODEL.DEEPLAB.NUM_CLASSES
        self.in_features              = cfg.MODEL.DEEPLAB.IN_FEATURES
        # fmt: on
        self.backbone = build_backbone(cfg,parent=self,*args,**kwargs)
        self.head = build_semantic_head(cfg.MODEL.DEEPLAB.HEAD_NAME,cfg=cfg.MODEL.DEEPLAB,
                                  parent=self,
                                  *args,**kwargs)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (H, W, C) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        batched_inputs = self.preprocess_image(batched_inputs)

        features = self.backbone(batched_inputs)
        if len(self.in_features) == 0:
            print(f"Error no input features for deeplab, use all features {features.keys()}")
            features = list(features.values())
        else:
            features = [features[f] for f in self.in_features]
        pred_logits = self.head(features)
        gt_labels = batched_inputs.get(GT_SEMANTIC_LABELS,None)

        outputs = build_outputs(name=self.cfg.MODEL.DEEPLAB.OUTPUTS,
            cfg=self.cfg.MODEL.DEEPLAB,
            parent=self,
            pred_logits=pred_logits,
            labels=gt_labels,
        )
        outputs.batched_inputs = batched_inputs
        max_outputs = 3
        wsummary.batch_semantic_summary(batched_inputs[IMAGE],masks=gt_labels[...,1:],max_outputs=max_outputs,name="gt")

        if self.is_training:
            if self.cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
                results = outputs.inference(inputs=batched_inputs,logits=pred_logits)
                wsummary.batch_semantic_summary(batched_inputs[IMAGE], masks=results[RD_SEMANTIC][...,1:],
                                                max_outputs=max_outputs,
                                                name="pred")
                wsummary.feature_map_summary(gt_labels, name="gt_semantic", max_outputs=10)
                wsummary.feature_map_summary(results[RD_SEMANTIC],name="pred_semantic", max_outputs=10)
            else:
                results = {}

            return results,outputs.losses()
        else:
            results = outputs.inference(inputs=batched_inputs,logits=pred_logits)
            wsummary.batch_semantic_summary(batched_inputs[IMAGE],
                                            masks=results[RD_SEMANTIC][...,1:],
                                            max_outputs=max_outputs,
                                            name="pred")
            wsummary.feature_map_summary(gt_labels, name="gt_semantic", max_outputs=10)
            wsummary.feature_map_summary(results[RD_SEMANTIC], name="pred_semantic", max_outputs=10)
            return results,{}
    
    def doeval(self,evaler,datas):
        assert datas[GT_BOXES].shape[0]==1,"Error batch size"
        kwargs = {}
        gt_labels = np.argmax(datas[GT_SEMANTIC_LABELS][0],axis=-1)
        pred_labels = datas[RD_SPARSE_SEMANTIC][0]

        kwargs['gtlabels'] = gt_labels
        kwargs['predictions'] = pred_labels

        evaler(**kwargs)
