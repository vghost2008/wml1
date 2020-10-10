#coding=utf-8
import tensorflow as tf
from .build import META_ARCH_REGISTRY
from object_detection2.modeling.bbdnet.build import build_bbdnet
from object_detection2.modeling.build import build_outputs
from object_detection2.modeling.backbone.build import build_backbone
from object_detection2.modeling.anchor_generator import build_anchor_generator
from object_detection2.modeling.box_regression import Box2BoxTransform
from object_detection2.modeling.build_matcher import build_matcher
from object_detection2.standard_names import *
from object_detection2.modeling.onestage_heads.retinanet_outputs import *
from .meta_arch import MetaArch
from object_detection2.datadef import *
from object_detection2.modeling.onestage_heads.build import build_retinanet_head

slim = tf.contrib.slim

__all__ = ["BBDRetinaNet"]

@META_ARCH_REGISTRY.register()
class BBDRetinaNet(MetaArch):
    """
    Implement RetinaNet (https://arxiv.org/abs/1708.02002).
    """

    def __init__(self, cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)

        # fmt: off
        self.num_classes              = cfg.MODEL.RETINANET.NUM_CLASSES
        self.in_features              = cfg.MODEL.RETINANET.IN_FEATURES
        # fmt: on

        self.backbone = build_backbone(cfg,parent=self,*args,**kwargs)

        self.anchor_generator = build_anchor_generator(cfg,parent=self,*args,**kwargs)
        self.head = build_retinanet_head(cfg.MODEL.RETINANET.HEAD_NAME,cfg=cfg.MODEL.RETINANET,
                                  num_anchors=self.anchor_generator.num_cell_anchors,
                                  parent=self,
                                  *args,**kwargs)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = build_matcher(
            cfg.MODEL.RETINANET.MATCHER,
            thresholds=cfg.MODEL.RETINANET.IOU_THRESHOLDS,
            allow_low_quality_matches=True,
            cfg=cfg,
            parent=self,
            k = self.anchor_generator.num_cell_anchors[0],
        )

    @wmlt.add_name_scope
    def merge_features(self,features):
        tower_nets = [features[0]]
        shape = wmlt.combined_static_and_dynamic_shape(tower_nets[0])
        for net in features[1:]:
            tower_nets.append(tf.image.resize_bilinear(net,shape[1:3]))
        tower_nets = tf.concat(tower_nets,axis=-1)
        return tower_nets

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

        bb_features = self.backbone(batched_inputs)
        if len(self.in_features) == 0:
            print(f"Error no input features for retinanet, use all features {bb_features.keys()}")
            features = list(bb_features.values())
        else:
            features = [bb_features[f] for f in self.in_features]
        pred_logits, pred_anchor_deltas= self.head(features)
        anchors = self.anchor_generator(batched_inputs,features)
        gt_boxes = batched_inputs.get(GT_BOXES,None)
        gt_length = batched_inputs.get(GT_LENGTH,None)
        gt_labels = batched_inputs.get(GT_LABELS,None)

        outputs = build_outputs(name=self.cfg.MODEL.RETINANET.OUTPUTS,
            cfg=self.cfg.MODEL.RETINANET,
            parent=self,
            box2box_transform=self.box2box_transform,
            anchor_matcher=self.anchor_matcher,
            pred_logits=pred_logits,
            pred_anchor_deltas=pred_anchor_deltas,
            anchors=anchors,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_length=gt_length,
            max_detections_per_image=self.cfg.TEST.DETECTIONS_PER_IMAGE,
        )
        outputs.batched_inputs = batched_inputs


        if hasattr(self.head,'logits_before_outputs'):
            map_data = tf.concat([self.head.logits_before_outputs,self.head.regs_before_outputs],axis=-1)
            print("Use direct map attr.")
        else:
            print("Use crop map attr.")
            if ";" in self.cfg.MODEL.BBDNET.MAP_DATA:
                keys = self.cfg.MODEL.BBDNET.MAP_DATA.split(";")
                map_data = [bb_features[x] for x in keys]
            else:
                keys = self.cfg.MODEL.BBDNET.MAP_DATA
                map_data = bb_features[keys]
        bbd_net_input = {}
        bbd_net_input['net_data'] = map_data
        bbd_net_input['base_net'] = features[-1]
        tower_nets0 = self.merge_features(self.head.logits_pre_outputs)
        tower_nets1 = self.merge_features(self.head.bbox_reg_pre_outputs)
        tower_nets = tf.concat([tower_nets0,tower_nets1],axis=-1)
        bbd_net_input[IMAGE] = batched_inputs[IMAGE]
        bbd_net_input['tower_nets'] = tower_nets
        bbd_net = build_bbdnet(self.cfg.MODEL.BBDNET.NAME,
                               num_classes=self.cfg.MODEL.RETINANET.NUM_CLASSES,cfg=self.cfg,parent=self,
                               threshold=0.02)
        loss = {}
        if self.is_training:
            results = outputs.inference(inputs=batched_inputs,box_cls=pred_logits,
                                            box_delta=pred_anchor_deltas, anchors=anchors,
                                            output_fix_nr=self.cfg.MODEL.BBDNET.BBOXES_NR)
            bbd_net_input.update(results)
            bbd_net_input[GT_BOXES] = batched_inputs[GT_BOXES]
            bbd_net_input[GT_LABELS] = batched_inputs[GT_LABELS]
            bbd_net_input[GT_LENGTH] = batched_inputs[GT_LENGTH]
            if self.cfg.MODEL.BBDNET.END2END_TRAIN:
                loss.update(outputs.losses())
            ###
            t_pred_logits = general_to_N_HWA_K_and_concat(pred_logits, K=self.num_classes)
            t_probs = tf.nn.sigmoid(t_pred_logits)
            t_probs = wmlt.batch_gather(t_probs, results[RD_INDICES])
            #t_probs = tf.Print(t_probs,["pl:",t_probs[1][33],results[RD_LABELS][1][33]],summarize=100)
            bbd_net_input[RD_RAW_PROBABILITY] = t_probs
            ###
            bbd_loss,bbd_outputs = bbd_net(bbd_net_input)
            results.update(bbd_outputs)
            loss.update(bbd_loss)

        else:
            results = outputs.inference(inputs=batched_inputs,box_cls=pred_logits,
                                        box_delta=pred_anchor_deltas, anchors=anchors)
            bbd_net_input.update(results)
            ###
            t_pred_logits = general_to_N_HWA_K_and_concat(pred_logits, K=self.num_classes)
            t_probs = tf.nn.sigmoid(t_pred_logits)
            t_probs = wmlt.batch_gather(t_probs, results[RD_INDICES])
            bbd_net_input[RD_RAW_PROBABILITY] = t_probs
            ###
            bbd_loss,bbd_outputs = bbd_net(bbd_net_input)
            results.update(bbd_outputs)

        if global_cfg.GLOBAL.SUMMARY_LEVEL <= SummaryLevel.DEBUG:
            wsummary.detection_image_summary(images=batched_inputs[IMAGE],
                                             boxes=results[RD_BOXES], classes=results[RD_LABELS],
                                             lengths=results[RD_LENGTH],
                                             scores=results[RD_PROBABILITY],
                                             name="BBDRetinaNet_result",
                                             category_index=DataLoader.category_index)
            return results,loss
