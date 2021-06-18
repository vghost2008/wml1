# coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
from wmodule import WModule
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
from object_detection2.modeling.box_regression import Box2BoxTransform
import object_detection2.config.config as config
from object_detection2.modeling.matcher import *
import wmodule
from object_detection2.modeling.anchor_generator import AnchorGeneratorF


class WMLTest(tf.test.TestCase):
    def testLevelsPooler(self):
        with self.test_session() as sess:
            num_level = 3
            features = []
            t_features = []
            for i in range(num_level):
                fea = np.ones([1,10,10,1])*i
                features.append(fea)
                t_features.append(np.ones([1,2,2,1])*(2-i))
            t_features = np.concatenate(t_features,axis=0)
            boxes = [[[0.0,0.0,0.59,0.9],[0.0,0.0,0.3,0.3],[0.0,0.0,0.1,0.1]]]
            img_size = [224,224]
            cfg = config.get_cfg()
            config.set_global_cfg(cfg)
            cfg = cfg.MODEL.ROI_BOX_HEAD
            cfg.canonical_box_size = 0.3*224
            cfg.canonical_level = 1
            p = wmodule.WModule(cfg)
            pooler = ROIPooler(cfg,parent=p,output_size=[2,2])
            features = pooler.forward(features,tf.convert_to_tensor(boxes,dtype=tf.float32),img_size=img_size)
            features = sess.run(features)
            self.assertAllClose(t_features,features,atol=1e-3)

    def testBox2BoxTransform(self):
        with self.test_session() as sess:
            # 人工核算
            np_gboxes = np.array(
                [[0.0, 0.0, 0.2, 0.2], [0.3, 0.3, 0.5, 0.6], [0.1, 0.1, 0.4, 0.4], [0.7, 0.7, 0.9, 0.8]]);
            np_labels = np.array([0,1, 2, 3, 0,4])
            np_boxes = np.array(
                [[0.0, 0.0, 0.2, 0.1], [0.0, 0.0, 0.2, 0.2], [0.101, 0.1, 0.44, 0.4], [0.73, 0.71, 0.91, 0.81],
                 [0.7, 0.1, 0.9, 0.5], [0.3, 0.481, 0.5, 0.7]]);
            gboxes = tf.constant(np_gboxes, dtype=tf.float32)
            labels = tf.constant(np_labels);
            boxes = tf.constant(np_boxes, dtype=tf.float32)

            trans = Box2BoxTransform(weights=[10, 10, 5, 5])
            indices = np.array([[0, 0, 2, 3, 0, 1]])

            out_boxes = trans.get_deltas(tf.expand_dims(boxes, 0),
                                         tf.expand_dims(gboxes, 0),
                                         tf.expand_dims(labels, 0),
                                         indices)
            r_boxes = sess.run(out_boxes)
            target_out_boxes = np.array([[[0,0,0,0],
                                          [0., 0., 0., 0.],
                                          [-0.6047199, 0., -0.61108774, 0.],
                                          [-1.1111111, -1., 0.5268025782891318, 0.],
                                          [0., 0., 0., 0.],
                                          [0., -6.4155245, 0., 1.573553724198501]]])
            self.assertAllClose(a=target_out_boxes, b=r_boxes, atol=1e-4, rtol=0.)

    def testBox2BoxTransform2(self):
        with self.test_session() as sess:
            #人工核算
            np_boxes = np.array([[0.0, 0.0, 0.2, 0.1], [0.0, 0.0, 0.2, 0.2], [0.101, 0.1, 0.44, 0.4], [0.73, 0.71, 0.91, 0.81],
                                 [0.7, 0.1, 0.9, 0.5], [0.3, 0.481, 0.5, 0.7]])
            boxes = tf.constant(np_boxes,dtype=tf.float32)
            trans = Box2BoxTransform(weights=[10, 10, 5, 5])
            target_out_boxes = tf.constant(np.array([[0.,0.,0.,0.],
                                          [0.,0.,0.,0.],
                                          [-0.6047199,0.,-0.61108774,0.],
                                          [-1.1111111,-1.,0.5268025782891318,0.],
                                          [0.,0.,0.,0.],
                                          [0.,-6.4155245,0.,1.573553724198501]]),dtype=tf.float32)
            target_out_scores = np.array([0.,1.,0.87941164,0.67400867,0.,0.29750007])
            target_out_remove_indices = np.array([True,False,False,False,False,False])
            keep_indices = tf.logical_and(tf.logical_not(target_out_remove_indices),tf.greater(target_out_scores,0.1))
            boxes = tf.boolean_mask(boxes,keep_indices)
            out_boxes = tf.boolean_mask(target_out_boxes,keep_indices)

            with tf.device("/gpu:0"):
                new_boxes = trans.apply_deltas(boxes=boxes,deltas=out_boxes)
            out_new_boxes= new_boxes.eval()
            target_new_boxes = np.array([[0.,0.,0.2,0.2],
                                         [0.09999999,0.09999999,0.4,0.4],
                                         [0.6999999,0.7,0.9,0.8],
                                         [0.3,0.3,0.5,0.6]])
            self.assertAllClose(a=out_new_boxes,b=target_new_boxes,atol=1e-5,rtol=0.)

    def testATSSMatcher(self):
        with self.test_session() as sess:
            parent = WModule(cfg=None)
            matcher = ATSSMatcher(k=2, cfg=None, parent=parent)
            boxes0 = tf.constant(np.array([[0, 0, 1, 1], [0.5, 0.5, 1, 1], [0.1, 0.1, 0.2, 0.2], [0, 0, 1, 0.5],
                                           [0.1, 0.3, 0.5, 0.6], [0.4, 0.5, 0.7, 0.8]]))
            boxes0 = tf.expand_dims(boxes0, axis=0)
            boxes1 = tf.constant(np.array([[-0.1, 0, 0.3, 1], [0.5, 0.5, 1, 1]]))
            boxes1 = tf.expand_dims(boxes1, axis=0)
            labels = tf.constant([[1, 4]])
            length = tf.convert_to_tensor([2])
            labels, scores, indices = matcher(boxes0, boxes1, labels, length, [3, 3])
            target_scores = np.array([[0.,1.,0.,0.,0.,0.]],dtype=np.float32)
            target_labels = np.array([[0,4,0,0,0,0]],dtype=np.int32)
            target_indices = np.array([[-1, 1,-1,-1,-1,-1]],dtype=np.int32)
            labels, scores, indices = sess.run([labels, scores, indices])
            self.assertAllClose(scores,target_scores,atol=1e-4)
            self.assertAllEqual(labels,target_labels)
            self.assertAllEqual(indices,target_indices)

    def testATSSMatcher3(self):
        with self.test_session() as sess:
            parent = WModule(cfg=None)
            matcher = ATSSMatcher3(thresholds=[0.4,0.5], cfg=None, parent=parent)
            boxes0 = tf.constant(np.array([[0.5, 0.1, 1, 0.9], [0.5, 0.5, 1, 1], [0.1, 0.1, 0.2, 0.2], [0, 0, 1, 0.5],
                                           [0.1, 0.3, 0.5, 0.6], [0.4, 0.5, 0.7, 0.8]]),dtype=tf.float32)
            boxes0 = tf.expand_dims(boxes0, axis=0)
            boxes1 = tf.constant(np.array([[-0.1, 0, 0.3, 1], [0.5, 0.5, 1, 1]]),dtype=tf.float32)
            boxes1 = tf.expand_dims(boxes1, axis=0)
            labels = tf.constant([[1, 4]])
            length = tf.convert_to_tensor([2])
            labels, scores, indices = matcher(boxes0, boxes1, labels, length, [3, 3])
            target_scores = np.array([[0.44444444,1.,0.025,0.,0.13043478,0.21428571]],dtype=np.float32)
            target_labels = np.array([[-1,4,0,0,0,0]],dtype=np.int32)
            target_indices = np.array([[-1, 1,-1,-1,-1,-1]],dtype=np.int32)
            labels, scores, indices = sess.run([labels, scores, indices])
            self.assertAllClose(scores,target_scores,atol=1e-4)
            self.assertAllEqual(labels,target_labels)
            self.assertAllEqual(indices,target_indices)

    def testAnchorGenerator(self):
        with self.test_session() as sess:
            H,W = 5,7
            size = [2.5,3.5]
            scales = [1.1, 1.2, 1.8]
            aspect_ratios = [0.5, 1.0, 2.0]
            boxes0 = tfop.anchor_generator([H, W], size, scales=scales, aspect_ratios=aspect_ratios)
            ag = AnchorGeneratorF(scales=scales,aspect_ratios=aspect_ratios)
            boxes1 = ag(shape=[H,W],size=size)
            boxes0 = odb.to_cxyhw(boxes0)
            boxes1 = odb.to_cxyhw(boxes1)
            diff = boxes1-boxes0
            v = sess.run([diff,boxes0,boxes1])
            self.assertAllClose(v[1],v[2],atol=1e-5)





if __name__ == "__main__":
    np.random.seed(int(time.time()))
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s %(funcName)s:%(message)s',
                        datefmt="%H:%M:%S")
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.test.main()
