#coding=utf-8
from object_detection2.modeling.meta_arch.retinanet import RetinaNet as _RetinaNet
from object_detection2.modeling.onestage_heads.retinanet_outputs import RetinaNetOutputs as _RetinaNetOutputs
import wml_tfutils as wmlt
import functools
from object_detection2.standard_names import *
from object_detection2.datadef import *
import tfop
from object_detection2.modeling.build import HEAD_OUTPUTS

@HEAD_OUTPUTS.register()
class PGRetinaNetOutputs(_RetinaNetOutputs):
    def __init__(self,cfg,parent,*args,**kwargs):
        super().__init__(cfg,parent=parent,*args,**kwargs)

    def predict_proposals(self):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B), where B is box dimension (4 or 5).
        """
        with tf.name_scope("predict_proposals"):
            batch_size = self.pred_logits[0].get_shape().as_list()[0]
            num_cell_anchors = self.pred_logits[0].get_shape().as_list()[-1]
            box_dim = self.pred_anchor_deltas[0].get_shape().as_list()[-1]//num_cell_anchors
            pred_anchor_deltas = [tf.reshape(x,[batch_size,-1,box_dim]) for x in self.pred_anchor_deltas]
            pred_anchor_deltas = tf.concat(pred_anchor_deltas,axis=1)
            proposals = self.box2box_transform.apply_deltas(deltas=pred_anchor_deltas,boxes=self.anchors)
            return proposals

    def predict_objectness_logits(self):
        """
        Return objectness logits in the same format as the proposals returned by
        :meth:`predict_proposals`.

        Returns:
            pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A).
        """
        with tf.name_scope("predict_objectness_logits"):
            batch_size = self.pred_logits[0].get_shape().as_list()[0]
            pred_logits = [tf.reshape(x,[batch_size,-1]) for x in self.pred_logits]
            pred_logits = tf.concat(pred_logits,axis=1)
            return pred_logits
