#coding=utf-8
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

class ODTest(tf.test.TestCase):
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

    def test_loss_v1(self):
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
            logging.info(f"loss:{loss0},{loss1},{loss2}.")

    def test_loss_v2_0(self):
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
            self.assertAllClose(loss2,od_loss2,atol=1e-6,rtol=0)
            self.assertAllClose(pdivn,od_pdivn,atol=1e-6,rtol=0)
            logging.info(f"loss:{loss0},{loss1},{loss2}.")

    def test_loss_v2_1(self):
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
            self.assertAllClose(loss2,od_loss2,atol=1e-6,rtol=0)
            self.assertAllClose(pdivn,od_pdivn,atol=1e-6,rtol=0)
            logging.info(f"loss:{loss0},{loss1},{loss2}.")

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

if __name__ == "__main__":
    np.random.seed(int(time.time()))
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s %(funcName)s:%(message)s',
                        datefmt="%H:%M:%S")
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.test.main()
