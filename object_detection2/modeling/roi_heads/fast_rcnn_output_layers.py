#coding=utf-8
import tensorflow as tf
import wmodule
import wml_tfutils as wmlt
from collections import Iterable

slim = tf.contrib.slim

class FastRCNNOutputLayers(wmodule.WChildModule):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, cfg,parent, num_classes, cls_agnostic_bbox_reg=False, box_dim=4,**kwargs):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of classes include background classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super().__init__(cfg,parent=parent,**kwargs)
        self.num_classes = num_classes
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.box_dim = box_dim

    def forward(self, x,scope="BoxPredictor"):
        with tf.variable_scope(scope):
            if not isinstance(x,tf.Tensor) and isinstance(x,Iterable):
                if self.cfg.MODEL.ROI_HEADS.PRED_IOU:
                    assert len(x)==3, "error x length."
                else:
                    assert len(x) == 2, "error x length."

                def trans(net):
                    if len(net.get_shape()) > 2:
                        shape = wmlt.combined_static_and_dynamic_shape(net)
                        dim = 1
                        for x in shape[1:]:
                            dim *= x
                        return tf.reshape(net,[shape[0],dim])
                    else:
                        return net
                x = [trans(v) for v in x]
                scores = slim.fully_connected(x[0],self.num_classes+1,activation_fn=None,
                                              normalizer_fn=None,scope="cls_score")
                foreground_num_classes = self.num_classes
                num_bbox_reg_classes = 1 if self.cls_agnostic_bbox_reg else foreground_num_classes
                proposal_deltas = slim.fully_connected(x[1],self.box_dim*num_bbox_reg_classes,activation_fn=None,
                                                       normalizer_fn=None,scope="bbox_pred")
                if self.cfg.MODEL.ROI_HEADS.PRED_IOU:
                    iou_logits = slim.fully_connected(x[2],1,
                                                      activation_fn=None,
                                                      normalizer_fn=None,
                                                      scope="iou_pred")
            else:
                if len(x.get_shape()) > 2:
                    shape = wmlt.combined_static_and_dynamic_shape(x)
                    x = tf.reshape(x,[shape[0],-1])
                scores = slim.fully_connected(x,self.num_classes+1,activation_fn=None,
                                              normalizer_fn=None,scope="cls_score")
                foreground_num_classes = self.num_classes
                num_bbox_reg_classes = 1 if self.cls_agnostic_bbox_reg else foreground_num_classes
                proposal_deltas = slim.fully_connected(x,self.box_dim*num_bbox_reg_classes,activation_fn=None,
                                              normalizer_fn=None,scope="bbox_pred")
                if self.cfg.MODEL.ROI_HEADS.PRED_IOU:
                    iou_logits = slim.fully_connected(x,1,
                                                      activation_fn=None,
                                                      normalizer_fn=None,
                                                      scope="iou_pred")

            if self.cfg.MODEL.ROI_HEADS.PRED_IOU:
                return scores, proposal_deltas,iou_logits
            else:
                return scores, proposal_deltas
