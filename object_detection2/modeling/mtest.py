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


class WMLTest(tf.test.TestCase):
    def testLevelsPooler(self):
        with self.test_session() as sess:
            num_level = 3
            features = []
            t_features = []
            for i in range(num_level):
                fea = np.ones([1,10,10,1])*i
                features.append(fea)
                t_features.append(np.ones([1,2,2,1])*(2-i))
            t_features = np.concatenate(t_features,axis=0)
            boxes = [[[0.0,0.0,0.59,0.9],[0.0,0.0,0.3,0.3],[0.0,0.0,0.1,0.1]]]
            cfg = config.get_cfg()
            cfg = cfg.MODEL.ROI_BOX_HEAD
            cfg.canonical_box_size = 0.3
            cfg.canonical_level = 1
            p = wmodule.WModule(cfg)
            pooler = ROIPooler(cfg,parent=p,output_size=[2,2])
            features = pooler.forward(features,tf.convert_to_tensor(boxes,dtype=tf.float32))
            features = sess.run(features)
            self.assertAllClose(t_features,features,atol=1e-3)




if __name__ == "__main__":
    np.random.seed(int(time.time()))
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s %(funcName)s:%(message)s',
                        datefmt="%H:%M:%S")
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.test.main()
