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
from wml_utils import *
import object_detection.bboxes as bboxes
import img_utils as wmli

class WMLTest(tf.test.TestCase):
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

    def testSparseSoftmaxCrossEntropyWithLogitsFL(self):
        with self.test_session() as sess:
            shape = [2,3,4,5]
            tf.set_random_seed(int(time.time()))
            logits = tf.random_uniform(shape=shape,minval=-9.,maxval=9.,dtype=tf.float32)
            labels = tf.random_uniform(shape=shape[:-1],minval=0,maxval=shape[-1],dtype=tf.int32)
            loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
            loss2= wnn.sparse_softmax_cross_entropy_with_logits_FL(labels=labels,logits=logits,gamma=0.,alpha=None)
            t_loss1,t_loss2,t_labels = sess.run([loss1,loss2,labels])
            print(t_labels)
            self.assertAllClose(a=t_loss1,b=t_loss2,atol=0.01,rtol=0)

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

    def testSparseSoftmaxCrossEntropyWithLogitsAlphaBalancedFL(self):
        with self.test_session() as sess:
            shape = [2,3,4,5]
            tf.set_random_seed(int(time.time()))
            logits = tf.random_uniform(shape=shape,minval=-9.,maxval=9.,dtype=tf.float32)
            labels = tf.random_uniform(shape=shape[:-1],minval=0,maxval=shape[-1],dtype=tf.int32)
            loss1 = wnn.sparse_softmax_cross_entropy_with_logits_FL(labels=labels,logits=logits,gamma=0.,alpha="auto")
            loss2= wnn.sparse_softmax_cross_entropy_with_logits_alpha_balanced(labels=labels,logits=logits,alpha="auto")
            t_loss1,t_loss2,t_labels = sess.run([loss1,loss2,labels])
            print(t_labels)
            self.assertAllClose(a=t_loss1,b=t_loss2,atol=0.01,rtol=0)

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


if __name__ == "__main__":
    np.random.seed(int(time.time()))
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s %(funcName)s:%(message)s',
                        datefmt="%H:%M:%S")
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.test.main()
