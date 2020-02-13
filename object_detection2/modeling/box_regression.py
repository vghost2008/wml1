#coding=utf-8
import tensorflow as tf
import math
import wtfop.wtfop_ops as wop
import wml_tfutils as wmlt
from object_detection2.datadef import EncodedData

_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)

class Box2BoxTransform(object):
    def __init__(self,weights=[10,10,5,5],scale_clamp=_DEFAULT_SCALE_CLAMP):
        self.weights = weights
        self.scale_clamp = scale_clamp


    def get_deltas(self,boxes,gboxes,labels,indices):
        """
        the labels,indices is the output of matcher
        boxes:[batch_size,N,4]
        gboxes:[batch_size,M,4]
        labels:[batch_size,N]
        indices:[batch_size,N]
        output:
        [batch_size,N,4]
        """
        return wop.get_boxes_deltas(boxes=boxes,gboxes=gboxes,labels=labels,indices=indices,
                                    scale_weights=self.weights)

    def get_deltas_by_proposals_data(self,proposals:EncodedData):
        """
        the labels,indices is the output of matcher
        boxes:[batch_size,N,4]
        gboxes:[batch_size,M,4]
        labels:[batch_size,N]
        indices:[batch_size,N]
        output:
        [batch_size,N,4]
        """
        boxes = proposals.boxes
        gboxes = proposals.gt_boxes
        indices = proposals.indices
        labels = proposals.gt_object_logits
        return wop.get_boxes_deltas(boxes=boxes,gboxes=gboxes,labels=labels,indices=indices,
                                    scale_weights=self.weights)

    def apply_deltas(self,deltas,boxes):
        '''
        :param deltas: [batch_size,N,4]/[N,4]
        :param boxes: [batch_size,N,4]/[N.4]
        :return:
        '''
        B0 = boxes.get_shape().as_list()[0]
        B1 = deltas.get_shape().as_list()[0]
        assert len(deltas.get_shape()) == len(boxes.get_shape()), "deltas and boxes's dims must be equal."

        if len(deltas.get_shape()) == 2:
            return wop.decode_boxes1(boxes, deltas, prio_scaling=[1/x for x in self.weights])

        if B0==B1:
            return wmlt.static_or_dynamic_map_fn(lambda x:wop.decode_boxes1(x[0], x[1], prio_scaling=[1/x for x in self.weights]),
                                                 elems=[boxes,deltas],dtype=tf.float32,back_prop=False)
        elif B0==1 and B1>1:
            boxes = tf.squeeze(boxes,axis=0)
            return wmlt.static_or_dynamic_map_fn(lambda x:wop.decode_boxes1(boxes, x, prio_scaling=[1/x for x in self.weights]),
                                             elems=deltas,dtype=tf.float32,back_prop=False)
        else:
            raise Exception("Error batch size")
