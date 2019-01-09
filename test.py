#coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
import numpy as np
from functools import reduce
from threadtoolkit import *
import time
from wml_utils import *

class SquareTest(tf.test.TestCase):
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

    def testParForEach(self):
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
            self.assertAllEqual(resdata,target_data)



if __name__ == "__main__":
    tf.test.main()
