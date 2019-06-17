#coding=utf-8
from object_detection.fasterrcnn import *
from object_detection.ssd import *
from object_detection.wlayers import WROIAlign
import tensorflow as tf
import wml_tfutils as wmlt
import logging
import wml_utils as wmlu
import math
import numpy as np
from functools import reduce
from threadtoolkit import *
import time
import wnn
import object_detection.npod_toolkit as npodt
import object_detection.fasterrcnn as fasterrcnn
from wml_utils import *
import object_detection.bboxes as bboxes
import img_utils as wmli
from object_detection.losses import *
import logging
import wtfop.wtfop_ops as wop
from object_detection.ssd import SSD
os.environ['CUDA_VISIBLE_DEVICES'] = ''

class TestFasterRCNN(FasterRCNN):
    def __init__(self,*kargs,**kwargs):
        super().__init__(*kargs,**kwargs)

    def _rcnFeatureExtractor(self,net,reuse=False):
        channel = net.get_shape().as_list()[-1]
        return slim.conv2d(net,channel,[1,1],weights_initializer=tf.ones_initializer,biases_initializer=None)

class ODTest(tf.test.TestCase):
    def test_batch_bboxes_jaccard(self):
        with self.test_session() as sess:
            boxes0 = tf.convert_to_tensor(np.array([[0.0, 0.0, 0.2, 0.2], [0.3, 0.3, 0.5, 0.6], [0.1, 0.1, 0.4, 0.4], [0.7, 0.7, 0.9, 0.8]]))
            boxes1 = tf.convert_to_tensor(np.array([[0.0, 0.1, 0.2, 0.3], [0.3, 0.3, 0.5, 0.6], [0.1, 0.1, 0.4, 0.4], [0.91, 0.7, 1.0, 0.8]]))
            iou = bboxes.batch_bboxes_jaccard(boxes0,boxes1)
            print(iou.eval())

    def testGetAnchorBoxes1(self):
        data0 = bboxes.get_anchor_bboxes(shape=[4, 4], sizes=[0.1, 0.2], ratios=[0.5,1., 2.])
        data1 = bboxes.get_anchor_bboxesv2(shape=[4, 4], sizes=[0.1, 0.2], ratios=[0.5,1., 2.])
        self.assertAllClose(a=data0,b=data1,atol=0.001,rtol=0)

    def testGetAnchorBboxes(self):
        with self.test_session() as sess:
            shape = [3,2]
            scales = [0.01,0.02]
            ratios = [1.,2.]
            anchors = bboxes.get_anchor_bboxes(shape=shape,sizes=scales,ratios=ratios,is_area=True)
            anchors = np.reshape(anchors,[3,2,4,4])
            expected_anchors = [[[[0.126,0.189,0.207,0.311],[0.109,0.207,0.224,0.293],[0.109,0.163,0.224,0.337],[0.085,0.189,0.248,0.311]]
                                    ,[[0.126,0.689,0.207,0.811],[0.109,0.707,0.224,0.793],[0.109,0.663,0.224,0.837],[0.085,0.689,0.248,0.811]] ]
                ,[[[0.459,0.189,0.541,0.311],[0.442,0.207,0.558,0.293],[0.442,0.163,0.558,0.337],[0.418,0.189,0.582,0.311]]
                                    ,[[0.459,0.689,0.541,0.811],[0.442,0.707,0.558,0.793],[0.442,0.663,0.558,0.837],[0.418,0.689,0.582,0.811]] ]
                ,[[[0.793,0.189,0.874,0.311],[0.776,0.207,0.891,0.293],[0.776,0.163,0.891,0.337],[0.752,0.189,0.915,0.311]]
                                    ,[[0.793,0.689,0.874,0.811],[0.776,0.707,0.891,0.793],[0.776,0.663,0.891,0.837],[0.752,0.689,0.915,0.811]]]]
            #format = "{:.3f}"
            #wmlu.show_nparray(anchors,name="anchors",format=format)
            data = npodt.minmax_to_cyxsr(np.reshape(anchors,[-1,4]),h=math.sqrt(shape[0]/shape[1]),w=math.sqrt(shape[1]/shape[0]))
            data = np.reshape(data,[3,2,4,4])
            #wmlu.show_nparray(data,name="data",format=format)
            #e_data = npodt.minmax_to_cyxsr(np.reshape(expected_anchors,[-1,4]))
            #e_data = np.reshape(e_data,[2,2,4,4])
            #wmlu.show_nparray(e_data,name="e_data",format=format)

            self.assertAllClose(expected_anchors,anchors,atol=1e-3,rtol=0)

    def testGetAnchorBboxes2(self):
        with self.test_session() as sess:
            shape = [512,512]
            ratios = [0.5,1.0,2.0]
            scales = [0.25,0.5,1.0]
            anchors = bboxes.get_anchor_bboxes(shape=shape,sizes=scales,ratios=ratios,is_area=True)
            print(bboxes.to_cxysa(anchors))
            data = npodt.minmax_to_cyxsr(np.reshape(anchors,[-1,4]),h=math.sqrt(shape[0]/shape[1]),w=math.sqrt(shape[1]/shape[0]))
            data = np.reshape(data,[3,2,4,4])
            #wmlu.show_nparray(data,name="data",format=format)
            #e_data = npodt.minmax_to_cyxsr(np.reshape(expected_anchors,[-1,4]))
            #e_data = np.reshape(e_data,[2,2,4,4])
            #wmlu.show_nparray(e_data,name="e_data",format=format)

            #self.assertAllClose(expected_anchors,anchors,atol=1e-3,rtol=0)

    def testGetAnchorBboxes2(self):
        shape = [2,2]
        scales = [0.01,0.02]
        ratios = [1.,2.]
        anchors = bboxes.get_anchor_bboxesv2(shape=shape,sizes=scales,ratios=ratios,is_area=True)
        anchors = np.reshape(anchors,[2,2,4,4])
        expected_anchors = [[[[0.20000000298023224,0.20000000298023224,0.30000001192092896,0.30000001192092896],[0.1792893260717392,0.214644655585289,0.320710688829422,0.2853553295135498],[0.1792893260717392,0.1792893260717392,0.320710688829422,0.320710688829422],[0.15000000596046448,0.20000000298023224,0.3499999940395355,0.30000001192092896]]
                                ,[[0.20000000298023224,0.699999988079071,0.30000001192092896,0.800000011920929],[0.1792893260717392,0.7146446704864502,0.320710688829422,0.7853553295135498],[0.1792893260717392,0.6792893409729004,0.320710688829422,0.8207106590270996],[0.15000000596046448,0.699999988079071,0.3499999940395355,0.800000011920929]] ]
            ,[[[0.699999988079071,0.20000000298023224,0.800000011920929,0.30000001192092896],[0.6792893409729004,0.214644655585289,0.8207106590270996,0.2853553295135498],[0.6792893409729004,0.1792893260717392,0.8207106590270996,0.320710688829422],[0.6499999761581421,0.20000000298023224,0.8500000238418579,0.30000001192092896]]
                                ,[[0.699999988079071,0.699999988079071,0.800000011920929,0.800000011920929],[0.6792893409729004,0.7146446704864502,0.8207106590270996,0.7853553295135498],[0.6792893409729004,0.6792893409729004,0.8207106590270996,0.8207106590270996],[0.6499999761581421,0.699999988079071,0.8500000238418579,0.800000011920929]]]]
        #format = "{:.3f}"
        #wmlu.show_nparray(anchors,name="anchors",format=format)
        #data = npodt.minmax_to_cyxsr(np.reshape(anchors,[-1,4]))
        #data = np.reshape(data,[2,2,4,4])
        #wmlu.show_nparray(data,name="data",format=format)
        #e_data = npodt.minmax_to_cyxsr(np.reshape(expected_anchors,[-1,4]))
        #e_data = np.reshape(e_data,[2,2,4,4])
        #wmlu.show_nparray(e_data,name="e_data",format=format)

        self.assertAllClose(expected_anchors,anchors,atol=1e-6,rtol=0)

    def testGetAnchorBboxes3(self):
        shape = [256,256]
        scales = [0.01,0.02,0.9,1.0]
        ratios = [0.5,1.,2.]
        anchors0 = bboxes.get_anchor_bboxesv2(shape=shape,sizes=scales,ratios=ratios)
        anchors1 = bboxes.get_anchor_bboxes(shape=shape,sizes=scales,ratios=ratios)
        self.assertAllClose(anchors0,anchors1,atol=1e-6,rtol=0)

    def testGetAnchorBboxes4(self):
        shape = [2,2]
        scales = [0.01,0.02]
        ratios = [1.,2.]
        anchors = bboxes.get_anchor_bboxesv2(shape=shape,sizes=scales,ratios=ratios,is_area=True)
        anchors = np.reshape(anchors,[2,2,4,4])
        expected_anchors = [[[[0.20000000298023224,0.20000000298023224,0.30000001192092896,0.30000001192092896],[0.1792893260717392,0.214644655585289,0.320710688829422,0.2853553295135498],[0.1792893260717392,0.1792893260717392,0.320710688829422,0.320710688829422],[0.15000000596046448,0.20000000298023224,0.3499999940395355,0.30000001192092896]]
                                ,[[0.20000000298023224,0.699999988079071,0.30000001192092896,0.800000011920929],[0.1792893260717392,0.7146446704864502,0.320710688829422,0.7853553295135498],[0.1792893260717392,0.6792893409729004,0.320710688829422,0.8207106590270996],[0.15000000596046448,0.699999988079071,0.3499999940395355,0.800000011920929]] ]
            ,[[[0.699999988079071,0.20000000298023224,0.800000011920929,0.30000001192092896],[0.6792893409729004,0.214644655585289,0.8207106590270996,0.2853553295135498],[0.6792893409729004,0.1792893260717392,0.8207106590270996,0.320710688829422],[0.6499999761581421,0.20000000298023224,0.8500000238418579,0.30000001192092896]]
                                ,[[0.699999988079071,0.699999988079071,0.800000011920929,0.800000011920929],[0.6792893409729004,0.7146446704864502,0.8207106590270996,0.7853553295135498],[0.6792893409729004,0.6792893409729004,0.8207106590270996,0.8207106590270996],[0.6499999761581421,0.699999988079071,0.8500000238418579,0.800000011920929]]]]
        #format = "{:.3f}"
        #wmlu.show_nparray(anchors,name="anchors",format=format)
        #data = npodt.minmax_to_cyxsr(np.reshape(anchors,[-1,4]))
        #data = np.reshape(data,[2,2,4,4])
        #wmlu.show_nparray(data,name="data",format=format)
        #e_data = npodt.minmax_to_cyxsr(np.reshape(expected_anchors,[-1,4]))
        #e_data = np.reshape(e_data,[2,2,4,4])
        #wmlu.show_nparray(e_data,name="e_data",format=format)

        self.assertAllClose(expected_anchors,anchors,atol=1e-6,rtol=0)

    def testGetAnchorBboxes5(self):
        shape = [256,256]
        scales = [0.01,0.02,0.9,1.0]
        ratios = [0.5,1.,2.]
        with self.test_session() as sess:
            anchors0 = bboxes.get_anchor_bboxesv3(shape=shape,sizes=scales,ratios=ratios,is_area=False)
            anchors1 = wop.anchor_generator(shape=shape,size=[1,1],scales=scales,aspect_ratios=ratios).eval()
            self.assertAllClose(anchors0,anchors1,atol=1e-6,rtol=0)

    '''def test_loss_v1(self):
        batch_size = 8
        box_nr = 64
        num_classes = 2
        with self.test_session() as sess:
            gregs = tf.random_uniform(shape=[batch_size,box_nr,4],minval=-3.,maxval=3.,dtype=tf.float32,
                                      seed=int(time.time()))
            glabels = tf.random_uniform(shape=[batch_size,box_nr],minval=0,maxval=num_classes,dtype=tf.int32,seed=int(time.time()))
            classes_logits = tf.random_uniform(shape=[batch_size,box_nr,num_classes],minval=-10.,maxval=10.,dtype=tf.float32)
            bboxes_regs = tf.random_uniform(shape=[batch_size,box_nr,4],minval=-3.,maxval=3.,dtype=tf.float32)
            bboxes_remove_indices = tf.random_uniform(shape=[batch_size,box_nr],minval=0,maxval=4,dtype=tf.int32)
            bboxes_remove_indices = tf.cast(bboxes_remove_indices,tf.bool)

            loss0,loss1,loss2,pdivn = od_loss(gregs, glabels, classes_logits, bboxes_regs, num_classes, reg_loss_weight=3.,
                        bboxes_remove_indices=bboxes_remove_indices, scope="Loss",
                        classes_wise=False,
                        neg_multiplier=2.0,
                        scale=10.0,
                        use_focal_loss=False)
            cod_loss = ODLoss(num_classes=num_classes,reg_loss_weight=3.,classes_wise=False,neg_multiplier=2.,scale=10.)
            od_loss0,od_loss1,od_loss2,od_pdivn = cod_loss(gregs,glabels,classes_logits,bboxes_regs,
                                                           bboxes_remove_indices=bboxes_remove_indices)
            loss0, loss1, loss2, pdivn,od_loss0,od_loss1,od_loss2,od_pdivn = sess.run([loss0, loss1, loss2, pdivn,od_loss0,od_loss1,od_loss2,od_pdivn])
            self.assertAllClose(loss0,od_loss0,atol=1e-6,rtol=0)
            self.assertAllClose(loss1,od_loss1,atol=1e-6,rtol=0)
            self.assertAllClose(loss2,od_loss2,atol=1e-6,rtol=0)
            self.assertAllClose(pdivn,od_pdivn,atol=1e-6,rtol=0)
            logging.info(f"loss:{loss0},{loss1},{loss2}.")'''

    '''def test_loss_v2_0(self):
        batch_size = 8
        box_nr = 64
        num_classes = 2
        with self.test_session() as sess:
            gregs = tf.random_uniform(shape=[batch_size,box_nr,4],minval=-3.,maxval=3.,dtype=tf.float32,
                                      seed=int(time.time()))
            glabels = tf.random_uniform(shape=[batch_size,box_nr],minval=0,maxval=num_classes,dtype=tf.int32,seed=int(time.time()))
            classes_logits = tf.random_uniform(shape=[batch_size,box_nr,num_classes],minval=-10.,maxval=10.,dtype=tf.float32)
            bboxes_regs = tf.random_uniform(shape=[batch_size,box_nr,4],minval=-3.,maxval=3.,dtype=tf.float32)
            bboxes_remove_indices = tf.random_uniform(shape=[batch_size,box_nr],minval=0,maxval=3,dtype=tf.int32)
            bboxes_remove_indices = tf.cast(bboxes_remove_indices,tf.bool)
            scores = tf.ones_like(glabels,dtype=tf.float32)

            loss0,loss1,loss2,pdivn = od_lossv2(gregs, glabels, classes_logits, bboxes_regs, num_classes,
                                              reg_loss_weight=3.,
                                              bboxes_remove_indices=bboxes_remove_indices, scope="Loss",
                                              classes_wise=False,
                                              neg_multiplier=2.0,
                                              scores=scores,
                                              scale=10.0)
            cod_loss = ODLoss(num_classes=num_classes,reg_loss_weight=3.,classes_wise=False,neg_multiplier=2.,scale=10.)
            od_loss0,od_loss1,od_loss2,od_pdivn = cod_loss(gregs,glabels,classes_logits,bboxes_regs,
                                                           scores=scores,
                                                           bboxes_remove_indices=bboxes_remove_indices)
            loss0, loss1, loss2, pdivn,od_loss0,od_loss1,od_loss2,od_pdivn = sess.run([loss0, loss1, loss2, pdivn,od_loss0,od_loss1,od_loss2,od_pdivn])
            self.assertAllClose(loss0,od_loss0,atol=1e-6,rtol=0)
            self.assertAllClose(loss1,od_loss1,atol=1e-6,rtol=0)
            self.assertAllClose(pdivn,od_pdivn,atol=1e-6,rtol=0)
            logging.info(f"loss:{loss0},{loss1},{loss2}.")'''

    '''def test_loss_v2_1(self):
        batch_size = 8
        box_nr = 64
        num_classes = 2
        with self.test_session() as sess:
            gregs = tf.random_uniform(shape=[batch_size,box_nr,4],minval=-3.,maxval=3.,dtype=tf.float32,
                                      seed=int(time.time()))
            glabels = tf.random_uniform(shape=[batch_size,box_nr],minval=0,maxval=num_classes,dtype=tf.int32,seed=int(time.time()))
            classes_logits = tf.random_uniform(shape=[batch_size,box_nr,num_classes],minval=-10.,maxval=10.,dtype=tf.float32)
            bboxes_regs = tf.random_uniform(shape=[batch_size,box_nr,4],minval=-3.,maxval=3.,dtype=tf.float32)
            bboxes_remove_indices = tf.random_uniform(shape=[batch_size,box_nr],minval=0,maxval=3,dtype=tf.int32)
            bboxes_remove_indices = tf.cast(bboxes_remove_indices,tf.bool)
            scores = tf.random_uniform(shape=[batch_size,box_nr],minval=0.5,maxval=1.,dtype=tf.float32,seed=int(time.time()))
            loss0,loss1,loss2,pdivn = od_lossv2(gregs, glabels, classes_logits, bboxes_regs, num_classes,
                                              reg_loss_weight=3.,
                                              bboxes_remove_indices=bboxes_remove_indices, scope="Loss",
                                              classes_wise=False,
                                              neg_multiplier=2.0,
                                              scores=scores,
                                              scale=10.0)
            cod_loss = ODLoss(num_classes=num_classes,reg_loss_weight=3.,classes_wise=False,neg_multiplier=2.,scale=10.)
            od_loss0,od_loss1,od_loss2,od_pdivn = cod_loss(gregs,glabels,classes_logits,bboxes_regs,
                                                           scores=scores,
                                                           bboxes_remove_indices=bboxes_remove_indices)
            loss0, loss1, loss2, pdivn,od_loss0,od_loss1,od_loss2,od_pdivn = sess.run([loss0, loss1, loss2, pdivn,od_loss0,od_loss1,od_loss2,od_pdivn])
            self.assertAllClose(loss0,od_loss0,atol=1e-6,rtol=0)
            self.assertAllClose(loss1,od_loss1,atol=1e-6,rtol=0)
            self.assertAllClose(pdivn,od_pdivn,atol=1e-6,rtol=0)
            logging.info(f"loss:{loss0},{loss1},{loss2}.")'''

    '''def test_loss_v3(self):
        batch_size = 8
        box_nr = 64
        num_classes = 2
        with self.test_session() as sess:
            gregs = tf.random_uniform(shape=[batch_size, box_nr, 4], minval=-3., maxval=3., dtype=tf.float32,
                                      seed=int(time.time()))
            glabels = tf.random_uniform(shape=[batch_size, box_nr], minval=0, maxval=num_classes, dtype=tf.int32,
                                        seed=int(time.time()))
            classes_logits = tf.random_uniform(shape=[batch_size, box_nr, num_classes], minval=-10., maxval=10.,
                                               dtype=tf.float32)
            bboxes_regs = tf.random_uniform(shape=[batch_size, box_nr, 4], minval=-3., maxval=3., dtype=tf.float32)
            bboxes_remove_indices = tf.random_uniform(shape=[batch_size, box_nr], minval=0, maxval=4,
                                                      dtype=tf.int32)
            bboxes_remove_indices = tf.cast(bboxes_remove_indices, tf.bool)

            loss0, loss1, loss2, pdivn = od_loss(gregs, glabels, classes_logits, bboxes_regs, num_classes,
                                                 reg_loss_weight=3.,
                                                 bboxes_remove_indices=bboxes_remove_indices, scope="Loss",
                                                 classes_wise=False,
                                                 neg_multiplier=2.0,
                                                 scale=10.0,
                                                 use_focal_loss=True)
            cod_loss = ODLossWithFocalLoss(gamma=2.0,alpha="auto",max_alpha_scale=10.0,
                                           num_classes=num_classes, reg_loss_weight=3., classes_wise=False, neg_multiplier=2.,
                              scale=10.)
            od_loss0, od_loss1, od_loss2, od_pdivn = cod_loss(gregs, glabels, classes_logits, bboxes_regs,
                                                              bboxes_remove_indices=bboxes_remove_indices)
            loss0, loss1, loss2, pdivn, od_loss0, od_loss1, od_loss2, od_pdivn = sess.run(
                [loss0, loss1, loss2, pdivn, od_loss0, od_loss1, od_loss2, od_pdivn])
            self.assertAllClose(loss0, od_loss0, atol=1e-6, rtol=0)
            self.assertAllClose(loss1, od_loss1, atol=1e-6, rtol=0)
            self.assertAllClose(loss2, od_loss2, atol=1e-6, rtol=0)
            self.assertAllClose(pdivn, od_pdivn, atol=1e-6, rtol=0)
            logging.info(f"loss:{loss0},{loss1},{loss2}.")'''
    def test_label_smooth_loss(self):
        batch_size = 8
        box_nr = 64
        num_classes = 2
        with self.test_session() as sess:
            gregs = tf.random_uniform(shape=[batch_size, box_nr, 4], minval=-3., maxval=3., dtype=tf.float32,
                                      seed=int(time.time()))
            glabels = tf.random_uniform(shape=[batch_size, box_nr], minval=0, maxval=num_classes, dtype=tf.int32,
                                        seed=int(time.time()))
            classes_logits = tf.random_uniform(shape=[batch_size, box_nr, num_classes], minval=-10., maxval=10.,
                                               dtype=tf.float32)
            bboxes_regs = tf.random_uniform(shape=[batch_size, box_nr, 4], minval=-3., maxval=3., dtype=tf.float32)
            bboxes_remove_indices = tf.random_uniform(shape=[batch_size, box_nr], minval=0, maxval=4,
                                                      dtype=tf.int32)
            bboxes_remove_indices = tf.cast(bboxes_remove_indices, tf.bool)

            od_loss = ODLoss(num_classes=num_classes, reg_loss_weight=3., classes_wise=False, neg_multiplier=2.,
                             scale=10.)
            loss0, loss1, loss2, pdivn = od_loss(gregs, glabels, classes_logits, bboxes_regs,
                                                              bboxes_remove_indices=bboxes_remove_indices)
            cod_loss = ODLossWithLabelSmooth(smoothed_value=1.0,
                                           num_classes=num_classes, reg_loss_weight=3., classes_wise=False, neg_multiplier=2.,
                              scale=10.)
            od_loss0, od_loss1, od_loss2, od_pdivn = cod_loss(gregs, glabels, classes_logits, bboxes_regs,
                                                              bboxes_remove_indices=bboxes_remove_indices)
            loss0, loss1, loss2, pdivn, od_loss0, od_loss1, od_loss2, od_pdivn = sess.run(
                [loss0, loss1, loss2, pdivn, od_loss0, od_loss1, od_loss2, od_pdivn])
            self.assertAllClose(loss0, od_loss0, atol=1e-6, rtol=0)
            self.assertAllClose(loss1, od_loss1, atol=1e-6, rtol=0)
            self.assertAllClose(loss2, od_loss2, atol=1e-6, rtol=0)
            self.assertAllClose(pdivn, od_pdivn, atol=1e-6, rtol=0)
            logging.info(f"loss:{loss0},{loss1},{loss2}.")

    def test_label_smoothv1_loss(self):
        batch_size = 8
        box_nr = 64
        num_classes = 2
        with self.test_session() as sess:
            gregs = tf.random_uniform(shape=[batch_size, box_nr, 4], minval=-3., maxval=3., dtype=tf.float32,
                                      seed=int(time.time()))
            glabels = tf.random_uniform(shape=[batch_size, box_nr], minval=0, maxval=num_classes, dtype=tf.int32,
                                        seed=int(time.time()))
            classes_logits = tf.random_uniform(shape=[batch_size, box_nr, num_classes], minval=-10., maxval=10.,
                                               dtype=tf.float32)
            bboxes_regs = tf.random_uniform(shape=[batch_size, box_nr, 4], minval=-3., maxval=3., dtype=tf.float32)
            bboxes_remove_indices = tf.random_uniform(shape=[batch_size, box_nr], minval=0, maxval=4,
                                                      dtype=tf.int32)
            bboxes_remove_indices = tf.cast(bboxes_remove_indices, tf.bool)

            od_loss = ODLoss(num_classes=num_classes, reg_loss_weight=3., classes_wise=False, neg_multiplier=2.,
                             scale=10.)
            loss0, loss1, loss2, pdivn = od_loss(gregs, glabels, classes_logits, bboxes_regs,
                                                 bboxes_remove_indices=bboxes_remove_indices)
            cod_loss = ODLossWithLabelSmoothV1(smoothed_value=1.0,
                                             num_classes=num_classes, reg_loss_weight=3., classes_wise=False, neg_multiplier=2.,
                                             scale=10.)
            od_loss0, od_loss1, od_loss2, od_pdivn = cod_loss(gregs, glabels, classes_logits, bboxes_regs,
                                                              bboxes_remove_indices=bboxes_remove_indices)
            loss0, loss1, loss2, pdivn, od_loss0, od_loss1, od_loss2, od_pdivn = sess.run(
                [loss0, loss1, loss2, pdivn, od_loss0, od_loss1, od_loss2, od_pdivn])
            self.assertAllClose(loss0, od_loss0, atol=1e-6, rtol=0)
            self.assertAllClose(loss1, od_loss1, atol=1e-6, rtol=0)
            self.assertAllClose(loss2, od_loss2, atol=1e-6, rtol=0)
            self.assertAllClose(pdivn, od_pdivn, atol=1e-6, rtol=0)
            logging.info(f"loss:{loss0},{loss1},{loss2}.")

    def test_select_rcn_bboxes(self):
        bboxes = []
        labels = []
        for i in range(8):
            bboxes.append([i,i+1,i+3,i+4])
            labels.append(int(i>=4)*i)
        keep_indices = np.ones_like(labels).astype(np.bool)
        #keep_indices = np.zeros_like(labels).astype(np.bool)
        keep_indices[1] = False
        keep_indices[5] = False
        print(labels)
        with self.test_session() as sess:
            labels = tf.convert_to_tensor(labels)
            keep_indices = tf.convert_to_tensor(keep_indices)
            labels,indices = fasterrcnn.FasterRCNN.selectRCNBoxes(labels,keep_indices,4,3)
            bboxes = tf.gather(bboxes,indices)
            labels,indices,bboxes = sess.run([labels,indices,bboxes])

        print(labels,indices,bboxes)

    def test_faster_rcnn(self):
        expected_data = np.array([[[[3.0,3.0,3.0],[3.0,3.0,3.0],[4.5,4.5,4.5]]
        ,[[3.0,3.0,3.0],[3.0,3.0,3.0],[4.5,4.5,4.5]]
        ,[[6.0,6.0,6.0],[6.0,6.0,6.0],[7.5,7.5,7.5]]
        ]
        ,[[[9.0,9.0,9.0],[9.0,9.0,9.0],[10.5,10.5,10.5]]
        ,[[9.0,9.0,9.0],[9.0,9.0,9.0],[10.5,10.5,10.5]]
        ,[[9.0,9.0,9.0],[9.0,9.0,9.0],[10.5,10.5,10.5]]
        ]
        ,[[[6.0,6.0,6.0],[6.0,6.0,6.0],[6.0,6.0,6.0]]
        ,[[6.0,6.0,6.0],[6.0,6.0,6.0],[6.0,6.0,6.0]]
        ,[[9.0,9.0,9.0],[9.0,9.0,9.0],[9.0,9.0,9.0]]
        ]
        ,[[[12.0,12.0,12.0],[12.0,12.0,12.0],[12.0,12.0,12.0]]
        ,[[12.0,12.0,12.0],[12.0,12.0,12.0],[12.0,12.0,12.0]]
        ,[[12.0,12.0,12.0],[12.0,12.0,12.0],[12.0,12.0,12.0]]
        ]
        ,[[[3.0,3.0,3.0],[3.0,3.0,3.0],[3.0,3.0,3.0]]
        ,[[3.0,3.0,3.0],[3.0,3.0,3.0],[3.0,3.0,3.0]]
        ,[[3.0,3.0,3.0],[3.0,3.0,3.0],[3.0,3.0,3.0]]
        ]
        ,[[[9.0,9.0,9.0],[9.0,9.0,9.0],[9.0,9.0,9.0]]
        ,[[9.0,9.0,9.0],[9.0,9.0,9.0],[9.0,9.0,9.0]]
        ,[[9.0,9.0,9.0],[9.0,9.0,9.0],[9.0,9.0,9.0]]
        ]
        ,[[[6.0,6.0,6.0],[6.0,6.0,6.0],[6.0,6.0,6.0]]
        ,[[6.0,6.0,6.0],[6.0,6.0,6.0],[6.0,6.0,6.0]]
        ,[[6.0,6.0,6.0],[6.0,6.0,6.0],[6.0,6.0,6.0]]
        ]
        ,[[[12.0,12.0,12.0],[12.0,12.0,12.0],[12.0,12.0,12.0]]
        ,[[12.0,12.0,12.0],[12.0,12.0,12.0],[12.0,12.0,12.0]]
        ,[[12.0,12.0,12.0],[12.0,12.0,12.0],[12.0,12.0,12.0]]
        ]
        ,[[[6.0,6.0,6.0],[6.0,6.0,6.0],[9.0,9.0,9.0]]
        ,[[6.0,6.0,6.0],[6.0,6.0,6.0],[9.0,9.0,9.0]]
        ,[[12.0,12.0,12.0],[12.0,12.0,12.0],[15.0,15.0,15.0]]
        ]
        ,[[[18.0,18.0,18.0],[18.0,18.0,18.0],[21.0,21.0,21.0]]
        ,[[18.0,18.0,18.0],[18.0,18.0,18.0],[21.0,21.0,21.0]]
        ,[[18.0,18.0,18.0],[18.0,18.0,18.0],[21.0,21.0,21.0]]
        ]
        ,[[[12.0,12.0,12.0],[12.0,12.0,12.0],[12.0,12.0,12.0]]
        ,[[12.0,12.0,12.0],[12.0,12.0,12.0],[12.0,12.0,12.0]]
        ,[[18.0,18.0,18.0],[18.0,18.0,18.0],[18.0,18.0,18.0]]
        ]
        ,[[[24.0,24.0,24.0],[24.0,24.0,24.0],[24.0,24.0,24.0]]
        ,[[24.0,24.0,24.0],[24.0,24.0,24.0],[24.0,24.0,24.0]]
        ,[[24.0,24.0,24.0],[24.0,24.0,24.0],[24.0,24.0,24.0]]
        ]
        ,[[[6.0,6.0,6.0],[6.0,6.0,6.0],[6.0,6.0,6.0]]
        ,[[6.0,6.0,6.0],[6.0,6.0,6.0],[6.0,6.0,6.0]]
        ,[[6.0,6.0,6.0],[6.0,6.0,6.0],[6.0,6.0,6.0]]
        ]
        ,[[[18.0,18.0,18.0],[18.0,18.0,18.0],[18.0,18.0,18.0]]
        ,[[18.0,18.0,18.0],[18.0,18.0,18.0],[18.0,18.0,18.0]]
        ,[[18.0,18.0,18.0],[18.0,18.0,18.0],[18.0,18.0,18.0]]
        ]
        ,[[[12.0,12.0,12.0],[12.0,12.0,12.0],[12.0,12.0,12.0]]
        ,[[12.0,12.0,12.0],[12.0,12.0,12.0],[12.0,12.0,12.0]]
        ,[[12.0,12.0,12.0],[12.0,12.0,12.0],[12.0,12.0,12.0]]
        ]
        ,[[[24.0,24.0,24.0],[24.0,24.0,24.0],[24.0,24.0,24.0]]
        ,[[24.0,24.0,24.0],[24.0,24.0,24.0],[24.0,24.0,24.0]]
        ,[[24.0,24.0,24.0],[24.0,24.0,24.0],[24.0,24.0,24.0]]
        ]
        ])
        with self.test_session() as sess:
            d0 = tf.ones([4,4,3],dtype=tf.float32)
            d1 = d0*2.0
            d2 = d0*3.0
            d3 = d0*4.0

            data0 = tf.concat([d0,d1],axis=1)
            data1 = tf.concat([d2,d3],axis=1)
            data0 = tf.concat([data0,data1],axis=0)
            data1 = data0*2
            data = tf.stack([data0,data1],axis=0)
            boxes = np.array([
                               [0.0,0.0,0.5,0.5],[0.5,0.0,1.0,0.5],
                              [0.0,0.5,0.5,1.0],[0.5,0.5,1.0,1.0],
                                [0.1,0.1,0.4,0.4],[0.6,0.0,0.9,0.4],
                                [0.1,0.6,0.4,0.9],[0.6,0.6,0.9,0.9],
                              ],np.float32)
            boxes = tf.convert_to_tensor(boxes)
            boxes = tf.stack([boxes,boxes],axis=0)
            m = TestFasterRCNN(2,[4,4],batch_size=2)
            m.rcnFeatureExtractor(data,boxes,roipooling=WROIAlign(),bin_size=3,reuse=False)
            sess.run(tf.global_variables_initializer())
            predict = sess.run(m.ssbp_net)
            self.assertAllClose(predict,expected_data,1e-5,0.)

    def test_ssd_gen_anchor_boxes(self):
        with self.test_session() as sess:
            box_specs_list = [
                [(0.1, 1.0), (0.2, 2.0), (0.2, 0.5)],
                [(0.35, 1.0), (0.35, 2.0), (0.35, 0.5), (0.35, 3.0), (0.35, 0.3333333333333333),
                 (0.4183300132670378, 1.0)],
                [(0.5, 1.0), (0.5, 2.0), (0.5, 0.5), (0.5, 3.0), (0.5, 0.3333333333333333), (0.570087712549569, 1.0)],
                [(0.65, 1.0), (0.65, 2.0), (0.65, 0.5), (0.65, 3.0), (0.65, 0.3333333333333333),
                 (0.7211102550927979, 1.0)],
                [(0.8, 1.0), (0.8, 2.0), (0.8, 0.5), (0.8, 3.0), (0.8, 0.3333333333333333), (0.8717797887081347, 1.0)],
                [(0.95, 1.0), (0.95, 2.0), (0.95, 0.5), (0.95, 3.0), (0.95, 0.3333333333333333),
                 (0.9746794344808963, 1.0)],
            ]

            for corners in box_specs_list:
                datas = SSD.get_a_layer_anchors(corners,shape=[1,1],size=[1,1])
                datas = sess.run(datas)
                for anchor_corners_out in datas:
                    #wmlu.show_nparray(anchor_corners_out)
                    d = bboxes.to_cxysa(anchor_corners_out)
                    wmlu.show_list(d)

    def test_ssd_gen_anchor_boxes(self):
        with self.test_session() as sess:
            box_specs_list = [
                [(0.1, 1.0), (0.2, 2.0), (0.2, 0.5)],
                [(0.35, 1.0), (0.35, 2.0), (0.35, 0.5), (0.35, 3.0), (0.35, 0.3333333333333333),
                 (0.4183300132670378, 1.0)]
            ]

            for corners in box_specs_list:
                print("T")
                datas = SSD.get_a_layer_anchors(corners,shape=[3,3],size=[1,1])
                datas = sess.run(datas)
                for anchor_corners_out in datas:
                    #wmlu.show_nparray(anchor_corners_out)
                    d = bboxes.to_cxysa(anchor_corners_out)
                    print("A")
                    wmlu.show_list(d)
                    print("B")

    def test_ssd_get_boxes1(self):
        with self.test_session() as sess:
            num_classes = 20
            m = SSD(num_classes=num_classes,batch_size=1)
            boxes = np.array([[
                [0.0,0.0,0.5,0.5],[0.5,0.0,1.0,0.5],
                [0.0,0.5,0.5,1.0],[0.5,0.5,1.0,1.0],
                [0.1,0.1,0.4,0.4],[0.6,0.0,0.9,0.4],
                [0.1,0.6,0.4,0.9],[0.6,0.6,0.9,0.9],
            ]],np.float32)
            boxes_nr = 8
            labels = np.array(range(boxes_nr),dtype=np.int32)+10

            m.logits = tf.reshape(tf.one_hot(labels,depth=num_classes),[1,boxes_nr,num_classes])
            m.logits = tf.random_uniform(shape=[1,boxes_nr,num_classes])
            m.regs = tf.zeros([1,boxes_nr,4])
            m.anchors = tf.convert_to_tensor(boxes)
            bboxes,labels,probs,lens = m.getBoxesV1(k=3,threshold=0.)
            indices = m.indices
            bboxes,labels,probs,lens,indices = sess.run([bboxes,labels,probs,lens,indices])
            self.assertAllClose(bboxes[0],boxes[0][indices[0]])
            
    def test_ssd_get_boxes2(self):
        with self.test_session() as sess:
            num_classes = 20
            m = SSD(num_classes=num_classes,batch_size=1)
            boxes = np.array([[
                [0.0,0.0,0.5,0.5],[0.5,0.0,1.0,0.5],
                [0.0,0.5,0.5,1.0],[0.5,0.5,1.0,1.0],
                [0.1,0.1,0.4,0.4],[0.6,0.0,0.9,0.4],
                [0.1,0.6,0.4,0.9],[0.6,0.6,0.9,0.9],
            ]],np.float32)
            boxes_nr = 8
            labels = np.array(range(boxes_nr),dtype=np.int32)+10

            m.logits = tf.reshape(tf.one_hot(labels,depth=num_classes),[1,boxes_nr,num_classes])
            m.logits = tf.random_uniform(shape=[1,boxes_nr,num_classes])
            m.regs = tf.zeros([1,boxes_nr,4])
            m.anchors = tf.convert_to_tensor(boxes)
            bboxes,labels,probs,lens = m.getBoxesV2(k=3,threshold=0.)
            indices = m.indices
            bboxes,labels,probs,lens,indices = sess.run([bboxes,labels,probs,lens,indices])
            self.assertAllClose(bboxes[0],boxes[0][indices[0]])

    def test_ssd_get_boxes3(self):
        with self.test_session() as sess:
            num_classes = 20
            m = SSD(num_classes=num_classes,batch_size=1)
            boxes = np.array([[
                [0.0,0.0,0.5,0.5],[0.5,0.0,1.0,0.5],
                [0.0,0.5,0.5,1.0],[0.5,0.5,1.0,1.0],
                [0.1,0.1,0.4,0.4],[0.6,0.0,0.9,0.4],
                [0.1,0.6,0.4,0.9],[0.6,0.6,0.9,0.9],
            ]],np.float32)
            boxes_nr = 8
            labels = np.array(range(boxes_nr),dtype=np.int32)+10

            m.logits = tf.reshape(tf.one_hot(labels,depth=num_classes),[1,boxes_nr,num_classes])
            m.logits = tf.random_uniform(shape=[1,boxes_nr,num_classes])
            m.regs = tf.zeros([1,boxes_nr,4])
            m.anchors = tf.convert_to_tensor(boxes)
            bboxes,labels,probs,lens = m.getBoxesV3(k=2)
            indices = m.indices
            bboxes,labels,probs,lens,indices = sess.run([bboxes,labels,probs,lens,indices])
            self.assertAllClose(bboxes[0],boxes[0][indices[0]])
            wmlu.show_list(indices)
            wmlu.show_list(bboxes)


if __name__ == "__main__":
    np.random.seed(int(time.time()))
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s %(funcName)s:%(message)s',
                        datefmt="%H:%M:%S")
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.test.main()
