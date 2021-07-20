# coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
import wnnlayer as wnnl
import logging
import wml_utils as wmlu
import math
import numpy as np
from functools import reduce
from threadtoolkit import *
import time
import wnn
import object_detection.npod_toolkit as npodt
from wml_utils import *
import object_detection.bboxes as bboxes
import img_utils as wmli
from object_detection2.modeling.poolers import ROIPooler
import object_detection2.config.config as config
import wmodule
from object_detection2.modeling.roi_heads.roi_heads import ROIHeads
from object_detection2.modeling.roi_heads import mask_head
from object_detection2.standard_names import *
from object_detection2.datadef import *
#tf.enable_eager_execution()

@wmlt.add_name_scope
def mask_rcnn_loss_old(inputs,pred_mask_logits, proposals:EncodedData,fg_selection_mask,log=True):
    '''

    :param inputs:inputs[GT_MASKS] [batch_size,N,H,W]
    :param pred_mask_logits: [Y,H,W,C] C==1 if cls_anostic_mask else num_classes, H,W is the size of mask
       not the position in org image
    :param proposals:proposals.indices:[batch_size,M], proposals.boxes [batch_size,M],proposals.gt_object_logits:[batch_size,M]
    :param fg_selection_mask: [X]
    X = batch_size*M
    Y = tf.reduce_sum(fg_selection_mask)
    :return:
    '''
    cls_agnostic_mask = pred_mask_logits.get_shape().as_list()[-1] == 1
    total_num_masks,mask_H,mask_W,C  = wmlt.combined_static_and_dynamic_shape(pred_mask_logits)
    assert mask_H==mask_W, "Mask prediction must be square!"

    gt_masks = inputs[GT_MASKS] #[batch_size,N,H,W]

    with tf.device("/cpu:0"):
        #当输入图像分辨率很高时这里可能会消耗过多的GPU资源，因此改在CPU上执行
        batch_size,X,H,W = wmlt.combined_static_and_dynamic_shape(gt_masks)
        #background include in proposals, which's indices is -1
        gt_masks = wmlt.batch_gather(gt_masks,tf.nn.relu(proposals.indices))
        gt_masks = tf.reshape(gt_masks,[-1,H,W])
        gt_masks = tf.boolean_mask(gt_masks,fg_selection_mask)

    boxes = proposals.boxes
    batch_size,box_nr,box_dim = wmlt.combined_static_and_dynamic_shape(boxes)
    boxes = tf.reshape(boxes,[batch_size*box_nr,box_dim])
    boxes = tf.boolean_mask(boxes,fg_selection_mask)

    with tf.device("/cpu:0"):
        #当输入图像分辨率很高时这里可能会消耗过多的GPU资源，因此改在CPU上执行
        gt_masks = tf.expand_dims(gt_masks,axis=-1)
        croped_masks_gt_masks = wmlt.tf_crop_and_resize(gt_masks,boxes,[mask_H,mask_W])

    if not cls_agnostic_mask:
        gt_classes = proposals.gt_object_logits
        gt_classes = tf.reshape(gt_classes,[-1])
        gt_classes = tf.boolean_mask(gt_classes,fg_selection_mask)
        pred_mask_logits = tf.transpose(pred_mask_logits,[0,3,1,2])
        pred_mask_logits = wmlt.batch_gather(pred_mask_logits,gt_classes-1) #预测中不包含背景
        pred_mask_logits = tf.expand_dims(pred_mask_logits,axis=-1)

    mask_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=croped_masks_gt_masks,logits=pred_mask_logits)
    mask_loss = tf.reduce_mean(mask_loss)

    return mask_loss
    pass


class WMLTest(tf.test.TestCase):
    @staticmethod
    def get_random_box(w=1.0,h=1.0):
        min_y = random.random()-0.5
        min_x = random.random()-0.5
        max_y = min_y+random.random()*(1.0-min_y)
        max_x = min_x+random.random()*(1.0-min_x)
        return (min_y*h,min_x*w,max_y*h,max_x*w)
    @staticmethod
    def get_random_boxes(nr=10,w=1.0,h=1.0):
        res = []
        for _ in range(nr):
            res.append(WMLTest.get_random_box())
        return np.array(res)
    def testSampleProposals(self):
        with self.test_session() as sess:
            labels = tf.convert_to_tensor([1,-1,2,3,4,0,0,0])
            neg_nr = 3
            pos_nr = 5
            labels,indices = ROIHeads.sample_proposals(labels=labels,neg_nr=neg_nr,pos_nr=pos_nr)
            print(labels.eval(),indices.eval())
        pass

    def testMaskLoss0(self):
        with self.test_session() as sess:
            mask = np.ones([2,5,224,224],dtype=np.float32)
            indices = np.array([[0,1],[4,3]])
            boxes = tf.convert_to_tensor(np.array([[[0,0,0.5,0.5],[0,0,0.5,0.5]],[[0.1,0.1,0.5,0.5],[0.3,0.3,0.9,0.9]]]),
                                         dtype=tf.float32)
            pred_masks = tf.convert_to_tensor(np.ones([3,31,31,1],dtype=np.float32))
            fg_selection_mask = np.reshape(np.array([[True,True],[False,True]]),[-1])
            inputs = {}
            inputs[GT_MASKS] = tf.convert_to_tensor(mask)
            inputs[IMAGE] = np.zeros_like([2,224,224,3],dtype=np.float32)
            proposals = EncodedData(indices=indices,boxes=boxes)
            loss = mask_head.mask_rcnn_loss(inputs,pred_mask_logits=pred_masks,proposals=proposals,
                                            fg_selection_mask=fg_selection_mask,
                                            log=False)
            self.assertAllClose(a=loss.eval(),b=0.3132625,atol=1e-3)

    def testMaskLoss0(self):
        with self.test_session() as sess:
            mask = np.ones([2,5,224,224],dtype=np.float32)
            indices = np.array([[0,1],[4,3]])
            boxes = tf.convert_to_tensor(np.array([[[0,0,0.5,0.5],[0,0,0.5,0.5]],[[0.1,0.1,0.5,0.5],[0.3,0.3,0.9,0.9]]]),
                                         dtype=tf.float32)
            pred_masks = tf.convert_to_tensor(np.zeros([3,31,31,1],dtype=np.float32))
            fg_selection_mask = np.reshape(np.array([[True,True],[False,True]]),[-1])
            inputs = {}
            inputs[GT_MASKS] = tf.convert_to_tensor(mask)
            inputs[IMAGE] = np.zeros_like([2,224,224,3],dtype=np.float32)
            proposals = EncodedData(indices=indices,boxes=boxes)
            loss = mask_head.mask_rcnn_loss(inputs,pred_mask_logits=pred_masks,proposals=proposals,
                                            fg_selection_mask=fg_selection_mask,
                                            log=False)
            self.assertAllClose(a=loss.eval(),b=0.693146,atol=1e-3)

    def testMaskLossTime(self):
        with self.test_session() as sess:
            box_nr = 256
            mask = np.random.random([2,40,640,640])
            indices = np.random.randint(0,40,[2,box_nr])-1
            boxes = tf.stack([self.get_random_boxes(box_nr),self.get_random_boxes(box_nr)],axis=0)
            boxes = tf.cast(boxes,dtype=tf.float32)

            pred_masks = tf.convert_to_tensor(np.random.random([2*box_nr,31,31,1])*10.0,dtype=np.float32)
            fg_selection_mask = np.reshape(np.random.random([2,box_nr])>0.5,[-1])
            pred_masks = tf.boolean_mask(pred_masks,fg_selection_mask)
            inputs = {}
            inputs[GT_MASKS] = tf.convert_to_tensor(mask)
            inputs[IMAGE] = np.zeros_like([2,640,640,3],dtype=np.float32)
            proposals = EncodedData(indices=indices,boxes=boxes)
            loss0 = mask_rcnn_loss_old(inputs,pred_mask_logits=pred_masks,proposals=proposals,
                                            fg_selection_mask=fg_selection_mask,
                                            log=False)
            loss1 = mask_head.mask_rcnn_loss(inputs,pred_mask_logits=pred_masks,proposals=proposals,
                                      fg_selection_mask=fg_selection_mask,
                                      log=False)
            with wmlu.TimeThis():
                loss0 = loss0.eval()
            with wmlu.TimeThis():
                loss1 = loss1.eval()
            self.assertAllClose(a=loss0,b=3.550261,atol=1e-3)
            self.assertAllClose(a=loss1,b=3.550261,atol=1e-3)

if __name__ == "__main__":
    np.random.seed(int(time.time()))
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s %(funcName)s:%(message)s',
                        datefmt="%H:%M:%S")
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.test.main()
