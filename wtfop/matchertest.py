#coding=utf-8
import tensorflow as tf
from wtfop.wtfop_ops import boxes_soft_nms,crop_boxes,boxes_encode,decode_boxes1,boxes_encode1,int_hash,matcher
import wtfop.wtfop_ops as wop
import numpy as np
import random
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
slim = tf.contrib.slim

class WTFOPTest(tf.test.TestCase):

    @staticmethod
    def get_random_box(w=1.0,h=1.0):
        min_y = random.random()
        min_x = random.random()
        max_y = min_y+random.random()*(1.0-min_y)
        max_x = min_x+random.random()*(1.0-min_x)
        return (min_y*h,min_x*w,max_y*h,max_x*w)

    @staticmethod
    def get_random_boxes(nr=10,w=1.0,h=1.0):
        res = []
        for _ in range(nr):
            res.append(WTFOPTest.get_random_box())
        return np.array(res)

    def test_wpad(self):
        with self.test_session() as sess:
            a = tf.constant([1,2,3])
            a = wop.wpad(a,[0,4])
            t_a = np.array([1,2,3,3,2,1,3])
            b = tf.constant([1,2,3,4,5,6])
            b = wop.wpad(b,[0,-4])
            t_b = np.array([1,2,3,4,5,6])
            self.assertAllEqual(a.eval(),t_a)
            self.assertAllEqual(b.eval(),t_b)

    def testMatcher0(self):
        config = tf.ConfigProto()
           #config.gpu_options.per_process_gpu_memory_fraction = FLAGS.memory_fraction
        config.gpu_options.allow_growth = True
        x = tf.ones([1,4,4,2],dtype=tf.float32)
        x = slim.conv2d(x,2,[3,3])
        with self.test_session(config=config) as sess:
            g_nr = 1000
            a_nr = 1000
            max_overlap_as_pos = True
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
            out_boxes, out_labels, out_scores, out_remove_indices,indices = boxes_encode(boxes,
                                                                                 gboxes,
                                                                                 glabels,
                                                                                 length=lens,
                                                                                 pos_threshold=0.7,
                                                                                 neg_threshold=0.3,
                                                                                 prio_scaling=[0.1, 0.1, 0.2, 0.2],max_overlap_as_pos=max_overlap_as_pos)
            out_labels0, out_scores0, indices0= matcher(boxes,
                                                                                 gboxes,
                                                                                 glabels,
                                                                                 length=lens,
                                                                                 pos_threshold=0.7,
                                                                                 neg_threshold=0.3,
                                                                                 max_overlap_as_pos=max_overlap_as_pos)
            out_boxes0 = wop.get_boxes_deltas(boxes=boxes,gboxes=gboxes,labels=out_labels0,indices=indices0,scale_weights=[10,10,5,5])
            out_remove_indices0 = tf.equal(out_labels0,-1)
            cout_boxes, cout_labels, cout_scores, cout_remove_indices,cout_indices = sess.run([out_boxes, out_labels, out_scores, out_remove_indices,indices])
            gout_labels, gout_scores, gout_remove_indices,gout_indices,gout_boxes = sess.run([out_labels0, out_scores0, out_remove_indices0,indices0,out_boxes0])
            ckeep_indices = np.logical_not(cout_remove_indices)
            gkeep_indices = np.logical_not(gout_remove_indices)
            self.assertAllEqual(a=cout_remove_indices,b=gout_remove_indices)
            a = cout_indices[ckeep_indices]
            b = gout_indices[gkeep_indices]
            c = cout_scores[ckeep_indices]
            d = gout_scores[gkeep_indices]
            self.assertAllEqual(a=cout_indices,b=gout_indices)
            a = cout_labels[ckeep_indices]
            b = gout_labels[gkeep_indices]
            self.assertAllEqual(a=a,b=b)
            self.assertAllClose(a=cout_boxes,b=gout_boxes,atol=1e-4)


if __name__ == "__main__":
    random.seed(int(time.time()))
    tf.test.main()
