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
tf.enable_eager_execution()


class WMLTest(tf.test.TestCase):
    def testSampleProposals(self):
        with self.test_session() as sess:
            labels = tf.convert_to_tensor([1,-1,2,3,4,0,0,0])
            neg_nr = 3
            pos_nr = 5
            labels,indices = ROIHeads.sample_proposals(labels=labels,neg_nr=neg_nr,pos_nr=pos_nr)
            print(labels,indices)




if __name__ == "__main__":
    np.random.seed(int(time.time()))
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s %(funcName)s:%(message)s',
                        datefmt="%H:%M:%S")
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.test.main()
