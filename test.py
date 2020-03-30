#coding=utf-8
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
import basic_tftools as btf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


class WMLTest(tf.test.TestCase):
    def testLabelSmoothV1(self):
        with self.test_session():
            labels = tf.constant([1, 2, 3, 4, 0, 2], dtype=tf.int32)
            v = wmlt.label_smoothv1(labels, 5, 0.9)
            a = v.eval()
            b = [[0.1,0.9,0.0,0.0,0.0],
                 [0.1,0.0,0.9,0.0,0.0],
                 [0.1,0.0,0.0,0.9,0.0],
                 [0.1,0.0,0.0,0.0,0.9],
                 [1.0,0.0,0.0,0.0,0.0],
                 [0.1,0.0,0.9,0.0,0.0]]
            self.assertAllClose(a,b,atol=0.0001,rtol=0)
    '''def testLabelSmooth(self):
        with self.test_session():
            labels = tf.constant([1, 2, 3, 4, 0, 2], dtype=tf.int32)
            v = wmlt.label_smooth(labels, 5, 0.9)
            a = v.eval()
            b = [[0.025,0.9,0.025,0.025,0.025],
                 [0.025,0.025,0.9,0.025,0.025],
                 [0.025,0.025,0.025,0.9,0.025],
                 [0.025,0.025,0.025,0.025,0.9],
                 [0.9,0.025,0.025,0.025,0.025],
                 [0.025,0.025,0.9,0.025,0.025]]
            self.assertAllClose(a,b,atol=0.0001,rtol=0)

    def testProbabilityCase(self):
        with self.test_session():
            prob_fn_pairs = [
                (0.1,lambda :tf.constant(1,dtype=tf.int32)),
            (0.2,lambda :tf.constant(2,dtype=tf.int32)),
            (0.7,lambda :tf.constant(3,dtype=tf.int32)) ]
            r = wmlt.probability_case(prob_fn_pairs)
            counter={1:0,2:0,3:0}
            for _ in range(10000):
                v = r.eval()
                counter[v] = counter[v]+1
            total_nr = reduce(lambda x,y:x+y,list(counter.values()))
            b = []
            for k in [1,2,3]:
                b.append(counter[k]/total_nr)
            print(b)

            self.assertAllClose(a=[0.1,0.2,0.7],b=b,atol=0.01,rtol=0)'''

    '''def testParForEach(self):
        data = list(range(9))
        target_data = []
        def fn(x,do_sleep=True):
            if do_sleep:
                time.sleep(3)
            return x*x
        for d in data:
            target_data.append(fn(d,False))
        target_data.sort()

        with TimeThis():
            resdata = par_for_each(data,fn,thread_nr=len(data))
            resdata.sort()
            self.assertAllEqual(resdata,target_data)'''

    '''def testSparseSoftmaxCrossEntropyWithLogitsFL(self):
        with self.test_session() as sess:
            shape = [2,3,4,5]
            tf.set_random_seed(int(time.time()))
            logits = tf.random_uniform(shape=shape,minval=-9.,maxval=9.,dtype=tf.float32)
            labels = tf.random_uniform(shape=shape[:-1],minval=0,maxval=shape[-1],dtype=tf.int32)
            loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
            loss2= wnn.sparse_softmax_cross_entropy_with_logits_FL(labels=labels,logits=logits,gamma=0.,alpha=None)
            t_loss1,t_loss2,t_labels = sess.run([loss1,loss2,labels])
            print(t_labels)
            self.assertAllClose(a=t_loss1,b=t_loss2,atol=0.01,rtol=0)'''

    def testSparseSoftmaxCrossEntropyWithLogitsAlphaBalanced(self):
        with self.test_session() as sess:
            shape = [2,3,4,5]
            tf.set_random_seed(int(time.time()))
            logits = tf.random_uniform(shape=shape,minval=-9.,maxval=9.,dtype=tf.float32)
            labels = tf.random_uniform(shape=shape[:-1],minval=0,maxval=shape[-1],dtype=tf.int32)
            loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
            loss2= wnn.sparse_softmax_cross_entropy_with_logits_alpha_balanced(labels=labels,logits=logits,alpha=None)
            t_loss1,t_loss2,t_labels = sess.run([loss1,loss2,labels])
            print(t_labels)
            self.assertAllClose(a=t_loss1,b=t_loss2,atol=0.01,rtol=0)

    '''def testSparseSoftmaxCrossEntropyWithLogitsAlphaBalancedFL(self):
        with self.test_session() as sess:
            shape = [2,3,4,5]
            tf.set_random_seed(int(time.time()))
            logits = tf.random_uniform(shape=shape,minval=-9.,maxval=9.,dtype=tf.float32)
            labels = tf.random_uniform(shape=shape[:-1],minval=0,maxval=shape[-1],dtype=tf.int32)
            loss1 = wnn.sparse_softmax_cross_entropy_with_logits_FL(labels=labels,logits=logits,gamma=0.,alpha=None)
            loss2= wnn.sparse_softmax_cross_entropy_with_logits_alpha_balanced(labels=labels,logits=logits,alpha=None)
            t_loss1,t_loss2,t_labels = sess.run([loss1,loss2,labels])
            print(t_labels)
            self.assertAllClose(a=t_loss1,b=t_loss2,atol=0.01,rtol=0)'''

    def testHierarchicalSparseSoftmaxCrossEntropy(self):
        with self.test_session() as sess:
            batch_size = 16
            N = 11
            Y = 99
            np_logits = np.random.uniform(low=-4,high=4,size=[batch_size,N,Y])
            num_classes = np.random.uniform(low=2,high=Y,size=[N]).astype(np.int32)
            np_labels = []
            for _ in range(batch_size):
                f_classes = np.random.uniform(low=0,high=N,size=()).astype(np.int32)
                s_classes = np.random.uniform(low=0,high=num_classes[f_classes],size=()).astype(np.int32)
                np_labels.append([f_classes,s_classes])

            np_labels = np.array(np_labels)
            logits = tf.convert_to_tensor(np_logits,dtype=tf.float32)
            labels = tf.convert_to_tensor(np_labels)
            loss = wnn.hierarchical_sparse_softmax_cross_entropy(logits=logits,labels=labels,num_classes=num_classes)
            loss = loss.eval()
            np_loss = []
            for i in range(batch_size):
                classes = np_labels[i]
                f_classes = classes[0]
                s_classes = classes[1]
                nr = num_classes[f_classes]
                l_logits = np_logits[i][f_classes][:nr]
                l_labels = s_classes
                l_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l_logits,labels=l_labels)
                np_loss.append(l_loss.eval())

            self.assertAllClose(a=loss,b=np_loss,atol=1e-3,rtol=0.)

    def testIndicesToDenseVector(self):
        with self.test_session() as sess:
            size = 100
            indices_nr = 20
            np_indices = np.random.randint(0,100,size=[indices_nr])
            np_data = np.zeros([size],dtype=np.int32)
            v = 1
            for indice in np_indices:
                np_data[indice] = v
            indices = tf.constant(np_indices,dtype=tf.int32)
            data = wmlt.indices_to_dense_vector(indices=indices,size=size,indices_value=v,dtype=tf.int32)
            data = data.eval()
            self.assertAllEqual(a=np_data,b=data)

    def testIndicesToDenseVector2(self):
        with self.test_session() as sess:
            size = 10
            np_indices = [1,3,3,4,7,9,2,1]
            np_data = [0,1,1,1,1,0,0,1,0,1]
            indices = tf.constant(np_indices,dtype=tf.int32)
            data = wmlt.indices_to_dense_vector(indices=indices,size=size,indices_value=1,dtype=tf.int32)
            data = data.eval()
            print(data)
            self.assertAllEqual(a=np_data,b=data)


    def test_select_2thdata_by_index(self):
        with self.test_session() as sess:
            data = [[1,2,3],[4,5,6],[7,8,9]]
            index = [1,2,0]
            res = [2,6,7]
            tf_res = wmlt.select_2thdata_by_index(data,index)
            self.assertAllEqual(res,tf_res.eval())

    def test_select_2thdata_by_index_v2(self):
        with self.test_session() as sess:
            data = [[1,2,3],[4,5,6],[7,8,9]]
            index = [1,2,0]
            res = [2,6,7]
            tf_res = wmlt.select_2thdata_by_index_v2(data,index)
            self.assertAllEqual(res,tf_res.eval())
    def test_select_2thdata_by_index_v3(self):
        with self.test_session() as sess:
            data = [[1,2,3],[4,5,6],[7,8,9]]
            index = [1,2,0]
            res = [2,6,7]
            tf_res = wmlt.select_2thdata_by_index_v3(data,index)
            self.assertAllEqual(res,tf_res.eval())

    def test_crop_image(self):
        with self.test_session() as sess:
            data = np.array(range(16))
            img = np.reshape(data,[4,4,1])
            res_imgs = [[0,1,4,5],
                        [2,3,6,7],
                        [8,9,12,13],
                        [10,11,14,15],
                        [5,6,9,10]]
            res_imgs = np.array(res_imgs)
            res_imgs = np.reshape(res_imgs,[5,2,2,1])
            imgs = wmli.crop_image(img,2,2)
            self.assertAllEqual(imgs.eval(),res_imgs)
            
    def test_mdict(self):
        mdict = MDict({"a":1,"b":2,"c":3})
        self.assertAllEqual([mdict.a,mdict.b,mdict.c],[1,2,3])

    def test_dropblock(self):
        with self.test_session() as sess:
            data = tf.ones(shape=[1,32,32,1],dtype=tf.float32)
            data =wnnl.dropblock(data,keep_prob=0.8,block_size=4,is_training=True)
            data = tf.reshape(data,shape=[32,32])
            data = tf.cast(data,tf.int32)
            wmlu.show_list(sess.run(data).tolist())

    def test_merge_mask(self):
        with self.test_session() as sess:
            mask0 = [True,False,True,True,True,False,False,True,True,True,False]
            mask1 = [True,True,True,False,False,True,False]
            res = [True,False,True,True,False,False,False,False,True,False,False]
            mask0 = tf.convert_to_tensor(mask0)
            mask1 = tf.convert_to_tensor(mask1)
            t_res = wmlt.merge_mask(mask0,mask1)
            t_res = sess.run(t_res)
            self.assertAllEqual(t_res,res)

    def test_mask_to_indices(self):
        with self.test_session() as sess:
            mask = [True, True, False, False, False, True]
            res = [0, 1, 5]
            mask = tf.convert_to_tensor(mask)
            t_res = wmlt.mask_to_indices(mask)
            t_res = sess.run(t_res)
            self.assertAllEqual(t_res,res)

    def test_batch_indices_to_mask(self):
        with self.test_session() as sess:
            data = [[1,4,5,6,7,9,0],[11,22,44,66,77,99,00]]
            indices = np.array([[1,2,5,0],[4,3,0,0]])
            lens = [3,2]
            size = 7
            r_mask = [[False,True,True,False,False,True,False],
                      [False,False,False,True,True,False,False]
                      ]
            r_indices = [[0,1,2,0],[1,0,0,0]]
            indices = tf.convert_to_tensor(indices)
            lens = tf.convert_to_tensor(lens)
            r_data = [[4,5,9,4],[77,66,66,66]]
            t_mask,t_indices = wmlt.batch_indices_to_mask(indices,lens,size)
            t_data = wmlt.batch_boolean_mask(data,t_mask,4)
            t_data = wmlt.batch_gather(t_data,t_indices)
            t_mask,t_indices,t_data = sess.run([t_mask,t_indices,t_data])
            self.assertAllEqual(t_mask,r_mask)
            self.assertAllEqual(t_indices,r_indices)
            self.assertAllEqual(t_data,r_data)

    def test_crop_and_resize(self):
        with self.test_session() as sess:
            image = np.array([np.zeros([10,10]),np.ones([10,10]),np.ones([10,10])*2])
            image = np.expand_dims(image,axis=-1).astype(np.float32)
            image = np.expand_dims(image,axis=0)
            bboxes = np.array([[0,0.3,0.1,0.4],[0.1,0.3,0.2,0.4],[0,0,1,1]]).astype(np.float32)
            bboxes = np.expand_dims(bboxes,axis=0)
            image = tf.convert_to_tensor(image)
            bboxes = tf.convert_to_tensor(bboxes)
            image = wmlt.tf_crop_and_resize(image,bboxes,[7,7])
            image = sess.run(image)
            d_image = np.array([np.zeros([7,7]),np.ones([7,7]),np.ones([7,7])*2])
            d_image = np.expand_dims(d_image,axis=-1)
            d_image = np.expand_dims(d_image,axis=0)
            self.assertAllClose(image,d_image,1e-5,0.0)

    def test_list_to_2dlist(self):
        a = [1,2,3,4,5,6,7,8,9]
        b = [[1,2],[3,4],[5,6],[7,8],[9,0]]
        c = wmlu.list_to_2dlist(a,2)
        print(c)
        c[-1].append(0)
        self.assertAllEqual(c,b)

    def test_orthogonal_regularizerv2(self):
        with self.test_session() as sess:
            print("test_orthogonal_regularizerv2")
            fn = wnnl.orthogonal_regularizerv2(1)
            weight = np.array(list(range(3)),np.float32)
            wmlu.show_list(weight)
            v = fn(weight)
            v = sess.run(v)
            self.assertAllClose(v,4.0,atol=1e-4)

    def test_orthogonal_regularizerv1(self):
        with self.test_session() as sess:
            print("test_orthogonal_regularizerv1")
            fn = wnnl.orthogonal_regularizer(1)
            weight = np.array(list(range(3)),np.float32)
            wmlu.show_list(weight)
            v = fn(weight)
            v = sess.run(v)
            self.assertAllClose(v,9,atol=1e-4)

    def test_channel_upsampling(self):
        with self.test_session() as sess:
            print("Test channel upsample")
            data_in = list(range(16))
            data_in = np.array(data_in)
            data_in = np.reshape(data_in,[1,2,2,4])
            expected_data = np.array([[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]])
            wmlu.show_list(data_in)
            data_in = tf.constant(data_in,dtype=tf.float32)
            data_in = wmlt.channel_upsample(data_in,scale=2)
            data_in = tf.squeeze(data_in,axis=-1)
            data_in = tf.squeeze(data_in,axis=0)
            data_out = sess.run(data_in)
            wmlu.show_list(data_out)
            self.assertAllClose(expected_data,data_out,atol=1e-4)

    def test_sort_data(self):
        with self.test_session() as sess:
            scores = [0.4,0.9,0.1,0.2]
            boxes = [[0.4,0.4,0.5,0.5],[0.9,0.9,1.0,1.0],[0.1,0.1,0.2,0.2],[0.2,0.2,0.3,0.3]]
            t_boxes = [[0.9,0.9,1.0,1.0],[0.4,0.4,0.5,0.5],[0.2,0.2,0.3,0.3],[0.1,0.1,0.2,0.2]]
            class_idxs = [4,9,1,2]
            scores = tf.convert_to_tensor(scores,dtype=tf.float32)
            boxes = tf.convert_to_tensor(boxes,dtype=tf.float32)
            class_idxs = tf.convert_to_tensor(class_idxs,tf.int32)
            x, y = wmlt.sort_data(key=scores, datas=[boxes, class_idxs])
            boxes, class_idxs= y
            scores,indices = x
            boxes,class_idxs,scores,indices = sess.run([boxes,class_idxs,scores,indices])
            self.assertAllEqual(indices,[1,0,3,2])
            self.assertAllClose(boxes,t_boxes,atol=1e-3)

    def test_twod_indexs_to_oned_indexs(self):
        with self.test_session() as sess:
            indexs = tf.convert_to_tensor([[0,1,3,5,6],[9,1,3,5,0],[1,4,5,6,2]])
            target = [0,1,3,5,6,19,11,13,15,10,21,24,25,26,22]
            indexs = btf.twod_indexs_to_oned_indexs(indexs,depth=10)
            indexs = indexs.eval()
            self.assertAllEqual(a=indexs,b=target)

    def test_deform_conv2d(self):
        with self.test_session() as sess:
            '''x = tf.ones([17,224,224,32])
            offset = tf.ones([17,224,224,3*3*2*8])
            out = wnnl.deform_conv2d(inputs=x,offset=offset,num_outputs=64,
                                     kernel_size=3,stride=1,num_groups=2,deformable_group=8)
            sess.run(tf.global_variables_initializer())
            print("test_deform_conv2d")
            print(out)
            print(out.eval())'''
            pass

    def test_top_k_mask(self):
        with self.test_session() as sess:
            a = tf.constant([9, 8, 0, 6, 7, 1])
            d = wmlt.top_k_mask(a, 4)
            sess.run(tf.global_variables_initializer())
            out = sess.run(d)
            self.assertAllEqual(out,[ True, True,False, True, True,False])
    def test_top_k_mask_nd(self):
        with self.test_session() as sess:
            a = tf.constant([[[9, 2, 3, 4, 19], [5, 6, 7, 8, 55], [11, 0, 9, 8, 0]]])
            d = wmlt.top_k_mask(a, 2)
            sess.run(tf.global_variables_initializer())
            out = sess.run(d)
            self.assertAllEqual(out,[[[ True,False,False,False, True],
                                     [False, False, False,  True,  True],
                                     [ True, False,  True, False, False]]])

if __name__ == "__main__":
    np.random.seed(int(time.time()))
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s %(funcName)s:%(message)s',
                        datefmt="%H:%M:%S")
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.test.main()
