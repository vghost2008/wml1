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
import walgorithm as wlg


class WMLTest(tf.test.TestCase):
    def test_list_to_2dlist(self):
        a = "a"
        b = "ac"
        dis = wlg.edit_distance(a,b)
        self.assertAllEqual(dis,1)
        a = "a1c"
        b = "acd"
        dis = wlg.edit_distance(a,b)
        self.assertAllEqual(dis,2)



if __name__ == "__main__":
    np.random.seed(int(time.time()))
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s %(funcName)s:%(message)s',
                        datefmt="%H:%M:%S")
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.test.main()
