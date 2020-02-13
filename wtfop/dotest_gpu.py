#coding=utf-8
import tensorflow as tf
from wtfop.wtfop_ops import boxes_soft_nms,crop_boxes,boxes_encode,decode_boxes1,boxes_encode1,int_hash
import wtfop.wtfop_ops as wop
import object_detection.npod_toolkit as npod
import numpy as np
import wml_utils as wmlu
import random
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class WTFOPTest(tf.test.TestCase):
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
            res.append(WTFOPTest.get_random_box())
        return np.array(res)

    '''def testEncodeBoxes(self):
        with self.test_session() as sess:
            with tf.device("/gpu:0"):
                #人工核算
                np_gboxes = np.array([[0.0, 0.0, 0.2, 0.2], [0.3, 0.3, 0.5, 0.6], [0.1, 0.1, 0.4, 0.4], [0.7, 0.7, 0.9, 0.8]]);
                np_labels = np.array([1,2,3,4])
                np_boxes = np.array([[0.0, 0.0, 0.2, 0.1], [0.0, 0.0, 0.2, 0.2], [0.101, 0.1, 0.44, 0.4], [0.73, 0.71, 0.91, 0.81],
                 [0.7, 0.1, 0.9, 0.5], [0.3, 0.481, 0.5, 0.7]]);
                np_lens = np.array([np_labels.shape[0]])
                gboxes = tf.constant(np_gboxes,dtype=tf.float32)
                glabels = tf.constant(np_labels);
                boxes = tf.constant(np_boxes,dtype=tf.float32)
                lens = tf.constant(np_lens,dtype=tf.int32)
                out_boxes, out_labels, out_scores, out_remove_indices,indices = boxes_encode(tf.expand_dims(boxes,0),
                                                                                     tf.expand_dims(gboxes,0),
                                                                                     tf.expand_dims(glabels,0),
                                                                                     length=lens,
                                                                                     pos_threshold=0.7,
                                                                                     neg_threshold=0.3,
                                                                                     prio_scaling=[0.1, 0.1, 0.2, 0.2])
                r_boxes,out_boxes, out_labels, out_scores, out_remove_indices,out_indices = sess.run([boxes,out_boxes, out_labels, out_scores, out_remove_indices,indices])
                target_out_boxes = np.array([[[0.,0.,0.,0.],
                      [0.,0.,0.,0.],
                      [-0.6047199,0.,-0.61108774,0.],
                      [-1.1111111,-1.,0.5268025782891318,0.],
                      [0.,0.,0.,0.],
                      [0.,-6.4155245,0.,1.573553724198501]]])
                target_out_labels = np.array([[0,1,3,4,0,2]])
                target_out_indices = np.array([[-1,0,2,3,-1,1]])
                target_out_scores = np.array([[0.,1.,0.87941164,0.67400867,0.,0.29750007]])
                target_out_remove_indices = np.array([[True,False,False,False,False,False]])
                self.assertAllEqual(a=target_out_remove_indices,b=out_remove_indices)
                self.assertAllEqual(a=target_out_indices,b=out_indices)
                self.assertAllClose(a=target_out_boxes,b=out_boxes,atol=1e-4,rtol=0.)
                self.assertAllEqual(a=target_out_labels,b=out_labels)
                self.assertAllClose(a=target_out_scores,b=out_scores,atol=1e-5,rtol=0.)
                self.assertAllClose(a=r_boxes,b=np_boxes,atol=1e-5,rtol=0.)

    def testEncodeBoxes2(self):
        with self.test_session() as sess:
            with tf.device("/gpu:0"):
                #人工核算
                np_gboxes = np.array([[0.0, 0.0, 0.2, 0.2], [0.3, 0.3, 0.5, 0.6], [0.1, 0.1, 0.4, 0.4], [0.7, 0.7, 0.9, 0.8]]);
                np_labels = np.array([100,200,300,400])
                np_boxes = np.array([[0.0, 0.0, 0.2, 0.1], [0.0, 0.0, 0.2, 0.2], [0.101, 0.1, 0.44, 0.4], [0.73, 0.71, 0.91, 0.81],
                 [0.7, 0.1, 0.9, 0.5], [0.3, 0.481, 0.5, 0.7]]);
                np_lens = np.array([np_labels.shape[0]])
                gboxes = tf.constant(np_gboxes,dtype=tf.float32)
                glabels = tf.constant(np_labels);
                boxes = tf.constant(np_boxes,dtype=tf.float32)
                lens = tf.constant(np_lens,dtype=tf.int32)
                out_boxes, out_labels, out_scores, out_remove_indices,indices = boxes_encode(tf.stack([boxes,boxes],axis=0),
                                                                                     tf.stack([gboxes,gboxes],axis=0),
                                                                                     tf.stack([glabels,glabels],axis=00),
                                                                                     length=tf.concat([lens,lens],axis=0),
                                                                                     pos_threshold=0.7,
                                                                                     neg_threshold=0.3,
                                                                                     prio_scaling=[0.1, 0.1, 0.2, 0.2])
                r_boxes,out_boxes, out_labels, out_scores, out_remove_indices,out_indices = sess.run([boxes,out_boxes, out_labels, out_scores, out_remove_indices,indices])
                target_out_boxes = np.array([[0.,0.,0.,0.],
                      [0.,0.,0.,0.],
                      [-0.6047199,0.,-0.61108774,0.],
                      [-1.1111111,-1.,0.5268025782891318,0.],
                      [0.,0.,0.,0.],
                      [0.,-6.4155245,0.,1.573553724198501]])
                target_out_labels = np.array([0,100,300,400,0,200])
                target_out_indices = np.array([-1,0,2,3,-1,1])
                target_out_scores = np.array([0.,1.,0.87941164,0.67400867,0.,0.29750007])
                target_out_remove_indices = np.array([True,False,False,False,False,False])
                self.assertAllEqual(a=target_out_remove_indices,b=out_remove_indices[0])
                self.assertAllEqual(a=target_out_remove_indices,b=out_remove_indices[1])
                self.assertAllEqual(a=target_out_indices,b=out_indices[0])
                self.assertAllEqual(a=target_out_indices,b=out_indices[1])
                self.assertAllClose(a=target_out_boxes,b=out_boxes[0],atol=1e-4,rtol=0.)
                self.assertAllClose(a=target_out_boxes,b=out_boxes[1],atol=1e-4,rtol=0.)
                self.assertAllEqual(a=target_out_labels,b=out_labels[0])
                self.assertAllEqual(a=target_out_labels,b=out_labels[1])
                self.assertAllClose(a=target_out_scores,b=out_scores[0],atol=1e-5,rtol=0.)
                self.assertAllClose(a=target_out_scores,b=out_scores[1],atol=1e-5,rtol=0.)
                self.assertAllClose(a=r_boxes,b=np_boxes,atol=1e-5,rtol=0.)

    def testEncodeBoxes1(self):
        with self.test_session() as sess:
            #人工核算
            np_gboxes = np.array([[0.0, 0.0, 0.2, 0.2], [0.3, 0.3, 0.5, 0.6], [0.1, 0.1, 0.4, 0.4], [0.7, 0.7, 0.9, 0.8]]);
            np_labels = np.array([1,2,3,4])
            np_boxes = np.array([[0.0, 0.0, 0.2, 0.1], [0.0, 0.0, 0.2, 0.2], [0.101, 0.1, 0.44, 0.4], [0.73, 0.71, 0.91, 0.81],
                                 [0.7, 0.1, 0.9, 0.5], [0.3, 0.481, 0.5, 0.7]]);
            #与上一个测试相比，长度不一样
            np_lens = np.array([np_labels.shape[0]-1])
            gboxes = tf.constant(np_gboxes,dtype=tf.float32)
            glabels = tf.constant(np_labels);
            boxes = tf.constant(np_boxes,dtype=tf.float32)
            lens = tf.constant(np_lens,dtype=tf.int32)
            out_boxes, out_labels, out_scores, out_remove_indices,indices = boxes_encode(tf.expand_dims(boxes,0),
                                                                                 tf.expand_dims(gboxes,0),
                                                                                 tf.expand_dims(glabels,0),
                                                                                 length=lens,
                                                                                 pos_threshold=0.7,
                                                                                 neg_threshold=0.3,
                                                                                 prio_scaling=[0.1, 0.1, 0.2, 0.2])
            out_boxes, out_labels, out_scores, out_remove_indices,out_indices = sess.run([out_boxes, out_labels, out_scores, out_remove_indices,indices])
            target_out_boxes = np.array([[[0.,0.,0.,0.],
                                          [0.,0.,0.,0.],
                                          [-0.6047199,0.,-0.61108774,0.],
                                          [0.,0.,0.,0.],
                                          [0.,0.,0.,0.],
                                          [0.,-6.4155245,0.,1.573553724198501]]])
            target_out_labels = np.array([[0,1,3,0,0,2]])
            target_out_indices = np.array([[-1,0,2,-1,-1,1]])
            target_out_scores = np.array([[0.,1.,0.87941164,0,0.,0.29750007]])
            target_out_remove_indices = np.array([[True,False,False,False,False,False]])
            self.assertAllEqual(a=target_out_indices,b=out_indices)
            self.assertAllEqual(a=target_out_remove_indices,b=out_remove_indices)
            self.assertAllClose(a=target_out_boxes,b=out_boxes,atol=1e-4,rtol=0.)
            self.assertAllEqual(a=target_out_labels,b=out_labels)
            self.assertAllClose(a=target_out_scores,b=out_scores,atol=1e-5,rtol=0.)

    def testDecodeBoxes(self):
        with self.test_session() as sess:
            #人工核算
            np_gboxes = np.array([[0.0, 0.0, 0.2, 0.2], [0.3, 0.3, 0.5, 0.6], [0.1, 0.1, 0.4, 0.4], [0.7, 0.7, 0.9, 0.8]]);
            np_labels = np.array([1,2,3,4])
            np_boxes = np.array([[0.0, 0.0, 0.2, 0.1], [0.0, 0.0, 0.2, 0.2], [0.101, 0.1, 0.44, 0.4], [0.73, 0.71, 0.91, 0.81],
                                 [0.7, 0.1, 0.9, 0.5], [0.3, 0.481, 0.5, 0.7]])
            gboxes = tf.constant(np_gboxes,dtype=tf.float32)
            glabels = tf.constant(np_labels);
            boxes = tf.constant(np_boxes,dtype=tf.float32)
            with tf.device("/gpu:0"):
                out_boxes, out_labels, out_scores, out_remove_indices = boxes_encode1(boxes,
                                                                                 gboxes,
                                                                                 glabels,
                                                                                 pos_threshold=0.7,
                                                                                 neg_threshold=0.3,
                                                                                 prio_scaling=[0.1, 0.1, 0.2, 0.2])
            out_boxes, out_labels, out_scores, out_remove_indices = sess.run([out_boxes, out_labels, out_scores, out_remove_indices])
            target_out_boxes = np.array([[0.,0.,0.,0.],
                                          [0.,0.,0.,0.],
                                          [-0.6047199,0.,-0.61108774,0.],
                                          [-1.1111111,-1.,0.5268025782891318,0.],
                                          [0.,0.,0.,0.],
                                          [0.,-6.4155245,0.,1.573553724198501]])
            target_out_labels = np.array([0,1,3,4,0,2])
            target_out_scores = np.array([0.,1.,0.87941164,0.67400867,0.,0.29750007])
            target_out_remove_indices = np.array([True,False,False,False,False,False])
            self.assertAllEqual(a=target_out_remove_indices,b=out_remove_indices)
            self.assertAllClose(a=target_out_boxes,b=out_boxes,atol=1e-4,rtol=0.)
            self.assertAllEqual(a=target_out_labels,b=out_labels)
            self.assertAllClose(a=target_out_scores,b=out_scores,atol=1e-5,rtol=0.)
            keep_indices = tf.logical_and(tf.logical_not(out_remove_indices),tf.greater(out_scores,0.1))
            boxes = tf.boolean_mask(boxes,keep_indices)
            out_boxes = tf.boolean_mask(out_boxes,keep_indices)

            with tf.device("/gpu:0"):
                new_boxes = decode_boxes1(boxes,out_boxes)
            out_new_boxes= new_boxes.eval()
            target_new_boxes = np.array([[0.,0.,0.2,0.2],
                                         [0.09999999,0.09999999,0.4,0.4],
                                         [0.6999999,0.7,0.9,0.8],
                                         [0.3,0.3,0.5,0.6]])
            self.assertAllClose(a=out_new_boxes,b=target_new_boxes,atol=1e-5,rtol=0.)'''

    def testEncodeBoxesSpeed(self):
        config = tf.ConfigProto()
           #config.gpu_options.per_process_gpu_memory_fraction = FLAGS.memory_fraction
        config.gpu_options.allow_growth = True
        with self.test_session(config=config) as sess:
            g_nr = 10000
            a_nr = 10000
            max_overlap_as_pos = False
            np_gboxes0 = self.get_random_boxes(g_nr)
            np_gboxes1 = self.get_random_boxes(g_nr)
            np_labels0 = np.random.randint(100,size=[g_nr])
            np_labels1 = np.random.randint(100,size=[g_nr])
            np_boxes0 = self.get_random_boxes(a_nr)
            np_boxes1 = self.get_random_boxes(a_nr)
            np_lens = np.array([np_labels0.shape[0],np_labels1.shape[0]])
            gboxes = tf.constant(np.stack([np_gboxes0,np_gboxes1],axis=0),dtype=tf.float32)
            glabels = tf.constant(np.stack([np_labels0,np_labels1],axis=0));
            boxes = tf.constant(np.stack([np_boxes0,np_boxes1],axis=0),dtype=tf.float32)
            lens = tf.constant(np_lens,dtype=tf.int32)
            with tf.device("/cpu:0"):
                out_boxes, out_labels, out_scores, out_remove_indices,indices = boxes_encode(boxes,
                                                                                 gboxes,
                                                                                 glabels,
                                                                                 length=lens,
                                                                                 pos_threshold=0.7,
                                                                                 neg_threshold=0.3,
                                                                                 prio_scaling=[0.1, 0.1, 0.2, 0.2],max_overlap_as_pos=max_overlap_as_pos)
            with tf.device("/gpu:0"):
                out_boxes0, out_labels0, out_scores0, out_remove_indices0,indices0 = boxes_encode(boxes,
                                                                                 gboxes,
                                                                                 glabels,
                                                                                 length=lens,
                                                                                 pos_threshold=0.7,
                                                                                 neg_threshold=0.3,
                                                                                 prio_scaling=[0.1, 0.1, 0.2, 0.2],max_overlap_as_pos=max_overlap_as_pos)
            with wmlu.TimeThis("CPU"):
                out_boxes, cout_labels, cout_scores, cout_remove_indices,cout_indices = sess.run([out_boxes, out_labels, out_scores, out_remove_indices,indices])
            with wmlu.TimeThis("GPU"):
                out_boxes, gout_labels, gout_scores, gout_remove_indices,gout_indices = sess.run([out_boxes0, out_labels0, out_scores0, out_remove_indices0,indices0])
            ckeep_indices = np.logical_not(cout_remove_indices)
            gkeep_indices = np.logical_not(gout_remove_indices)
            self.assertAllEqual(a=cout_remove_indices,b=gout_remove_indices)
            a = cout_indices[ckeep_indices]
            b = gout_indices[gkeep_indices]
            c = cout_scores[ckeep_indices]
            d = gout_scores[gkeep_indices]
            self.assertAllEqual(a=cout_indices,b=gout_indices)
            a = cout_labels
            b = gout_labels
            self.assertAllEqual(a=a,b=b)

    '''def testDecodeBBoxesSpeed(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with self.test_session(config=config) as sess:
            nr = 100000
            np_aboxes = self.get_random_boxes(nr)
            np_regs = self.get_random_boxes(nr)
            aboxes = tf.constant(np_aboxes,dtype=tf.float32)
            regs = tf.constant(np_regs,dtype=tf.float32)
            with tf.device("/gpu:0"):
                gout = decode_boxes1(aboxes,regs)
            with tf.device("/cpu:0"):
                cout = decode_boxes1(aboxes,regs)
            with wmlu.TimeThis("CPUDecode-----------"):
                cout = sess.run(cout)
            with wmlu.TimeThis("GPUDecode-----------"):
                gout = sess.run(gout)
            self.assertAllClose(a=cout,b=gout,atol=1e-4,rtol=0.)'''

if __name__ == "__main__":
    random.seed(int(time.time()))
    tf.test.main()
