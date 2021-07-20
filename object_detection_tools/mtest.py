# coding=utf-8
import tensorflow as tf
import logging
from object_detection_tools.get_bbox_sizes_and_ratios import *
import time


class WMLTest(tf.test.TestCase):
    def testGetBoxSizeAndRatiosV1(self):
        boxes = np.array([[0,0,1,1],[0.5,0.0,1,1],[0.0,0.5,1.0,1.0],[0,0,1,1]])
        sizes,ratios = get_bboxes_sizes_and_ratios_by_kmeans_v1(boxes,ratio_nr=3,size_nr=2)
        target_sizes = [0.71,1.00]
        target_ratios = [[0.50,1.00,2.00],[0.50,1.00,2.00]]
        self.assertAllClose(sizes,target_sizes,atol=1e-2)
        self.assertAllClose(ratios,target_ratios,atol=1e-2)

    def testGetBoxSizeAndRatiosV2(self):
        boxes = np.array([[0,0,1,1],[0.5,0.0,1,1],[0.0,0.5,1.0,1.0],[0,0,0.25,2],[0,0,0.5,2],[0,0,2,0.5]])
        sizes,ratios = get_bboxes_sizes_and_ratios_by_kmeans_v2(boxes,ratio_nr=3,size_nr=2)
        target_sizes = [0.71, 1.00]
        target_ratios = [[0.50, 2.00, 8.00], [0.25, 1.00, 4.00]]
        self.assertAllClose(sizes,target_sizes,atol=1e-2)
        self.assertAllClose(ratios,target_ratios,atol=1e-2)




if __name__ == "__main__":
    np.random.seed(int(time.time()))
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s %(funcName)s:%(message)s',
                        datefmt="%H:%M:%S")
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.test.main()
