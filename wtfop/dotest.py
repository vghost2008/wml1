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
        min_y = random.random()
        min_x = random.random()
        max_y = min_y+random.random()*(1.0-min_y)
        max_x = min_x+random.random()*(1.0-min_x)
        return (min_y*h,min_x*w,max_y*h,max_x*w)

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


    def testSoftNMS(self):
        with self.test_session() as sess:
            boxes=tf.constant([[124,60,251,153],[161,85,293,193],[104,103,266,222],[277,371,414,484]],dtype=tf.float32)
            labels = tf.constant([1,1,1,1],dtype=tf.int32)
            probs = tf.constant([0.9,0.8,0.8,0.9],dtype=tf.float32)
            boxes, labels, indices = boxes_soft_nms(boxes, labels, confidence=probs,
                                                    threshold=0.7,
                                                    classes_wise=True)
            boxes,labels,indices = sess.run([boxes,labels,indices])
            print(boxes)
            print(labels)
            print(indices)
            self.assertAllEqual(a=indices,b=[0,2,3],msg="index equall")
    def testCropBoxes(self):
        with self.test_session() as sess:
            #np_gboxes = np.array([[0.0, 0.0, 0.2, 0.2], [0.3, 0.3, 0.5, 0.6], [0.1, 0.1, 0.4, 0.4], [0.7, 0.7, 0.9, 0.8]])
            #np_subbox = np.array([0.1, 0.1, 0.7, 0.7])
            np_gboxes_nr = 128
            np_gboxes = []
            for _ in range(np_gboxes_nr):
                np_gboxes.append(self.get_random_box())
            np_gboxes = np.array(np_gboxes)
            np_subbox = self.get_random_box()
            np_res_boxes,np_res_mask = npod.crop_box(bboxes=np_gboxes, sub_box=np_subbox, remove_threshold=0.8)
            gboxes = tf.constant(np_gboxes, dtype=tf.float32);
            ref_boxes = tf.constant(np_subbox, dtype=tf.float32);
            boxes, mask = crop_boxes(ref_box=ref_boxes, boxes=gboxes, threshold=0.8)
            res_boxes,res_mask = sess.run([boxes,mask])
            self.assertAllEqual(a=np_res_mask,b=res_mask)
            self.assertAllClose(a=np_res_boxes,b=res_boxes,atol=1e-6,rtol=0.)

    def testEncodeBoxes(self):
        with self.test_session() as sess:
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
            out_boxes, out_labels, out_scores, out_remove_indices,out_indices = sess.run([out_boxes, out_labels, out_scores, out_remove_indices,indices])
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

    def test_adjacent_matrix_generator(self):
        np_gboxes = np.array([
            [0.0, 0.0, 0.2, 0.2], [0.1, 0.1, 0.5, 0.4], [0.1, 0.1, 0.4, 0.4], [0.1, 0.0, 0.2, 0.3],
            [0.5, 0.5, 0.6, 0.6], [0.5, 0.6, 0.7, 0.8], [0.5, 0.7, 0.6, 0.8], [0.7, 0.7, 0.9, 0.8],
        ]);
        with self.test_session() as sess:
            am = wop.adjacent_matrix_generator(bboxes=np_gboxes,theta=60.,scale=1)
            am = sess.run(am)
            print(np.sum(am))
            wmlu.show_nparray(am,"AM")
    '''def test_adjacent_matrix_generator_by_iou(self):
        with self.test_session() as sess:
            np_gboxes = np.array([
                [0.0, 0.0, 0.2, 0.2], [0.0, 0.1, 0.2, 0.3], [0.1, 0.1, 0.4, 0.4], [0.1, 0.0, 0.2, 0.3],
                [0.5, 0.5, 0.6, 0.6], [0.5, 0.5, 0.6, 0.62], [0.5, 0.7, 0.6, 0.8], [0.55, 0.7, 0.65, 0.8],
            ]);
        with self.test_session() as sess:
            am = wop.adjacent_matrix_generator_by_iou(bboxes=np_gboxes,threshold=0.3)
            self.assertAllEqual(am.eval(),[[0,1,0,1,0,0,0,0],[1,0,0,1,0,0,0,0],[0,0,0,1,1,0,0,0],[1,1,1,0,0,0,0,0],[0,0,1,0,0,1,0,0],[0,0,0,0,1,0,1,0],[0,0,0,0,0,1,0,1],[0,0,0,0,0,0,1,0]])
            bm = wop.adjacent_matrix_generator_by_iou(bboxes=np_gboxes,threshold=0.3,keep_connect=False)
            wmlu.show_nparray(bm.eval())
            self.assertAllEqual(bm.eval(),[[0,1,0,1,0,0,0,0],[1,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]])
            '''

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

            new_boxes = decode_boxes1(boxes,out_boxes)
            out_new_boxes= new_boxes.eval()
            target_new_boxes = np.array([[0.,0.,0.2,0.2],
                                         [0.09999999,0.09999999,0.4,0.4],
                                         [0.6999999,0.7,0.9,0.8],
                                         [0.3,0.3,0.5,0.6]])
            self.assertAllClose(a=out_new_boxes,b=target_new_boxes,atol=1e-5,rtol=0.)

    '''def testLabelType(self):
        text = []
        for i in range(ord('a'), ord('z') + 1):
            text.append(chr(i))
        for i in range(ord('A'), ord('Z') + 1):
            text.append(chr(i))
        for i in range(ord('0'), ord('9') + 1):
            text.append(chr(i))
        text.append('/')
        text.append('\\')
        text.append('-')
        text.append('+')
        text.append(":")
        text.append("WORD")
        text_to_id = {}
        for i, t in enumerate(text):
            text_to_id[t] = i + 1
        def string_to_ids(v):
            res = []
            for c in v:
                res.append(text_to_id[c])
            return res
        def make_bboxes(ids):
            w = 1.
            h  = 2.;
            res = []
            for i in range(len(ids)):
                res.append([0.,w*i,h,w*(i+1)])
            res.append([0.,0.,h,w*(len(ids))])
            return np.array(res)
        test_data=[
           "Ki-67","kI-67","ER","er","Her-2","HER-2","HP","hp",
            "k-67","eir","hr-","hhpp","89K-67","PHX80718"
        ]
        expected_data=[0,0,1,1,2,2,4,4,0,1,2,4,0,4]
        t_bboxes = tf.placeholder(dtype=tf.float32,shape=[None,4])
        t_labels = tf.placeholder(dtype=tf.int32,shape=[None])
        t_type = label_type(bboxes=t_bboxes,labels=t_labels)
        with self.test_session() as sess:
            for i,data in enumerate(test_data):
                print(i)
                ids = string_to_ids(data)
                bboxes = make_bboxes(ids)
                ids.append(68)
                type = sess.run(t_type,feed_dict={t_bboxes:bboxes,t_labels:ids})
                print(test_data[i],type)
                self.assertAllEqual(type,np.array([expected_data[i]]))'''

    def testIntHash(self):
        with self.test_session() as sess:
            key = [1,3,4,5,6,7,8]
            value = [3,3,9,1,10,99,88]
            table = dict(zip(key,value))
            with self.test_session() as sess:
                input = tf.constant(key)
                out = int_hash(input,table)
                out = sess.run(out)
                self.assertAllEqual(out,value)

    def testSampleLabels(self):
        print("Test sample labels")
        ids = tf.placeholder(dtype=tf.int32,shape=[2,None])
        labels = tf.placeholder(dtype=tf.int32,shape=[2,None])
        res = wop.sample_labels(labels=labels,ids=ids,sample_nr=8)
        with self.test_session() as sess:
            _ids = np.array([[0,0,0,0,0],[0,0,0,0,0]])
            _labels = np.ones_like(_ids);
            feed_dict = {ids:_ids,labels:_labels}
            r = sess.run(res,feed_dict=feed_dict)
            wmlu.show_list(r)
            _ids = np.array([[0,1,1,2,2],[1,0,0,1,0]])
            _labels = np.ones_like(_ids);
            feed_dict = {ids:_ids,labels:_labels}
            r = sess.run(res,feed_dict=feed_dict)
            wmlu.show_list(r)
            _ids = np.array([[1,1,3,3,2,2,2,0],[1,1,2,2,3,3,3,1]])
            _labels = np.array([[1,1,1,1,2,2,2,0],[1,1,1,1,3,3,3,1]])
            feed_dict = {ids:_ids,labels:_labels}
            r = sess.run(res,feed_dict=feed_dict)
            wmlu.show_list(r)

    def testBoxesMatch(self):
        with self.test_session() as sess:
            #人工核算
            np_gboxes = np.array([[[0.0, 0.0, 0.2, 0.2], [0.3, 0.3, 0.5, 0.6], [0.1, 0.1, 0.4, 0.4], [0.7, 0.7, 0.9, 0.8]]]);
            np_labels = np.array([[1,2,3,4]])
            np_boxes = np.array([[[0.0, 0.0, 0.2, 0.1], [0.0, 0.0, 0.2, 0.2], [0.101, 0.1, 0.44, 0.4], [0.73, 0.71, 0.91, 0.81],
                                 [0.7, 0.1, 0.9, 0.5], [0.3, 0.4, 0.5, 0.7],[0.3, 0.39, 0.5, 0.69]]]);
            np_lens = np.array([np_labels.shape[1]])
            gboxes = tf.constant(np_gboxes,dtype=tf.float32)
            glabels = tf.constant(np_labels);
            boxes = tf.constant(np_boxes,dtype=tf.float32)
            lens = tf.constant(np_lens,dtype=tf.int32)
            plabels = tf.constant([[0,1,0,0,0,2,0]],dtype=tf.int32)
            plabels1 = tf.constant([[0,2,0,0,0,1,0]],dtype=tf.int32)
            labels,scores = wop.boxes_match(boxes,gboxes,glabels,lens,threshold=0.5)
            self.assertAllEqual(labels.eval(),[[0,1,3,4,0,0,2]])
            labels, scores = wop.boxes_match_with_pred(boxes, plabels,gboxes, glabels, lens, threshold=0.5)
            self.assertAllEqual(labels.eval(),[[0,1,3,4,0,2,0]])
            labels, scores = wop.boxes_match_with_pred(boxes, plabels1,gboxes, glabels, lens, threshold=0.5)
            self.assertAllEqual(labels.eval(),[[0,1,3,4,0,0,2]])
            labels, scores,encode = wop.boxes_match_with_pred2(boxes, plabels1,gboxes, glabels, lens, threshold=0.5)
            self.assertAllEqual(labels.eval(),[[0,1,3,4,0,0,2]])
            np_encodes = np.array([[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[-0.604719877243042,0.0,-0.6110877394676208,0.0],
                                    [-1.111116647720337,-0.9999988079071045,0.5268022418022156,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,-2.999998092651367,0.0,0.0]]])
            self.assertAllClose(encode.eval(),np_encodes,atol=1e-6)

    def test_adjacent_matrix_generator_by_iou1(self):
        with self.test_session() as sess:
            np_bboxes = np.array([[0.0, 0.0, 0.2, 0.2], [0.01, 0.02, 0.2, 0.19], [0.5, 0.5, 0.6, 0.6], [0.49, 0.5, 0.61, 0.6]]);
            bboxes = tf.constant(np_bboxes,dtype=tf.float32)
            adj_mt = wop.adjacent_matrix_generator_by_iou(bboxes,0.5,False)
            self.assertAllEqual(adj_mt.eval(),[[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])
            adj_mt = wop.adjacent_matrix_generator_by_iou(bboxes,0.5,True)
            self.assertAllEqual(adj_mt.eval(),[[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]])

    def test_merge_line_bboxes(self):
        print("test merge line bboxes")
        with self.test_session() as sess:
            bboxes = np.array([[[0.0,0.0,0.01,0.5],[0.01,0.01,0.11,0.4],[0.4,0.0,0.41,0.5],[0.4,0.51,0.41,0.9]]])
            datas = np.array([[[0.0,0.0,0.00,0.0],[0.00,0.00,0.0,0.0],[0.0,0.0,0.00,0.0],[1.4,1.51,1.41,1.9]]])
            datas = np.array([[[0.0,0.0,0.00,0.0],[0.00,0.00,0.0,0.0],[0.0,0.0,0.00,0.0],[0.0,0.0,0.00,0.0]]])
            labels = np.array([[1,1,1,1]])
            lens = np.array([4])
            threshold = 0.5
            dis_threshold = [0.1,0.1]
            ids = wop.merge_line_boxes(data=datas,labels=labels,bboxes=bboxes,lens=lens,threshold=threshold,dis_threshold=dis_threshold)
            r_ids = sess.run(ids)
            print(r_ids)

    def test_center_boxes_decode(self):
        with self.test_session() as sess:
            print("test test center boxes decode")
            tl = np.zeros([1,9,9,3],dtype=np.float32)
            br = np.zeros([1,9,9,3],dtype=np.float32)
            c = np.zeros([1,9,9,3],dtype=np.float32)
            offset_tl = np.zeros([1,9,9,2],dtype=np.float32)
            offset_br = np.zeros([1,9,9,2],dtype=np.float32)
            offset_c = np.zeros([1,9,9,2],dtype=np.float32)
            tl[0,0,0,0] = 1
            br[0,2,2,0] = 1
            c[0,1,1,0] = 1

            tl[0,5,5,2] = 0.7
            br[0,7,7,2] = 0.8
            c[0,6,6,2] = 0.9
            bboxes,labels,probs,index,lens = wop.center_boxes_decode(heatmaps_tl=tl,heatmaps_br=br,heatmaps_c=c,offset_tl=offset_tl,
                                                               offset_br=offset_br,offset_c=offset_c,k=10)
            bboxes,labels,probs,index,lens = sess.run([bboxes,labels,probs,index,lens])
            t_bboxes = [[[0.0, 0.0, 0.25, 0.25],
              [0.625, 0.625, 0.875, 0.875]]
             ]
            t_labels = [[0, 2]]
            t_probs = [[1.0,0.800000011920929]]
            t_indexs = [[10, 60]]
            t_lens = [2]
            self.assertAllClose(bboxes,t_bboxes,atol=1e-5)
            self.assertAllClose(t_probs,probs,atol=1e-5)
            self.assertAllEqual(t_labels,labels)
            self.assertAllEqual(t_indexs,index)
            self.assertAllEqual(t_lens,lens)
            wmlu.show_nparray(bboxes)
            wmlu.show_nparray(labels)
            wmlu.show_nparray(probs)
            wmlu.show_nparray(index)
            wmlu.show_nparray(lens)

if __name__ == "__main__":
    random.seed(int(time.time()))
    tf.test.main()
