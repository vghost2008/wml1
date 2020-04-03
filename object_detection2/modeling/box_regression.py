#coding=utf-8
import tensorflow as tf
import math
import wtfop.wtfop_ops as wop
import wml_tfutils as wmlt
from object_detection2.datadef import EncodedData
import object_detection2.bboxes as odb

_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)
class AbstractBox2BoxTransform(object):
    def get_deltas(self,boxes,gboxes,labels,indices,img_size=None):
        """
        the labels,indices is the output of matcher
        boxes:[batch_size,N,4]
        gboxes:[batch_size,M,4]
        labels:[batch_size,N]
        indices:[batch_size,N]
        output:
        [batch_size,N,4]
        """
        pass

    def get_deltas_by_proposals_data(self,proposals:EncodedData,img_size=None):
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
        return self.get_deltas(boxes=boxes,gboxes=gboxes,labels=labels,indices=indices,img_size=img_size)

    def apply_deltas(self,deltas,boxes,img_size=None):
        '''
        :param deltas: [batch_size,N,4]/[N,4]
        :param boxes: [batch_size,N,4]/[N.4]
        :return:
        '''
        pass

class Box2BoxTransform(AbstractBox2BoxTransform):
    '''
    实现经典的Faster-RCN, RetinaNet中使用的编码解码方式
    '''
    def __init__(self,weights=[10,10,5,5],scale_clamp=_DEFAULT_SCALE_CLAMP):
        self.weights = weights
        self.scale_clamp = scale_clamp


    def get_deltas(self,boxes,gboxes,labels,indices,img_size=None):
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

    @staticmethod
    def decode_boxes(boxes,deltas,prio_scaling):
        #return wop.decode_boxes1(boxes=boxes,res=deltas,prio_scaling=prio_scaling)
        return odb.decode_boxes(boxes=boxes,regs=deltas,prio_scaling=prio_scaling)

    def apply_deltas(self,deltas,boxes,img_size=None):
        '''
        :param deltas: [batch_size,N,4]/[N,4]
        :param boxes: [batch_size,N,4]/[N.4]
        :return:
        '''
        B0 = boxes.get_shape().as_list()[0]
        B1 = deltas.get_shape().as_list()[0]
        assert len(deltas.get_shape()) == len(boxes.get_shape()), "deltas and boxes's dims must be equal."

        if len(deltas.get_shape()) == 2:
            return self.decode_boxes(boxes, deltas, prio_scaling=[1/x for x in self.weights])

        if B0==B1:
            return wmlt.static_or_dynamic_map_fn(lambda x:self.decode_boxes(x[0], x[1], prio_scaling=[1/x for x in self.weights]),
                                                 elems=[boxes,deltas],dtype=tf.float32,back_prop=False)
        elif B0==1 and B1>1:
            boxes = tf.squeeze(boxes,axis=0)
            return wmlt.static_or_dynamic_map_fn(lambda x:self.decode_boxes(boxes, x, prio_scaling=[1/x for x in self.weights]),
                                             elems=deltas,dtype=tf.float32,back_prop=False)
        else:
            raise Exception("Error batch size")

