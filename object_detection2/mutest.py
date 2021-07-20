# coding=utf-8
import tensorflow as tf
import logging
from wml_utils import *
import object_detection2.mask as odm


class WMLTest(tf.test.TestCase):
    def testMaskArea(self):
        with self.test_session() as sess:
            masks = [[0,1,1,0],[1,0,0,0],[0,0,0,0],[1,1,1,1]]
            boxes = np.array([[0,0,1,1],[0.5,0.5,1,1],[0.1,0.1,0.2,0.2],[0,0,1,1]])
            masks = tf.constant(masks,dtype=tf.uint8)
            masks = tf.reshape(masks,[4,2,2])
            boxes = tf.constant(boxes,dtype=tf.float32)
            size = [100,100]
            area = odm.mask_area_by_instance_mask(masks,boxes,size)
            sess.run(tf.global_variables_initializer())
            area = sess.run(area)
            self.assertAllClose(area,[ 5000. ,  625.,    0.,10000.],atol=1e-3)

    def testMaskArea2(self):
        with self.test_session() as sess:
            masks = [[0,1,1,0],[1,0,0,0],[0,0,0,0],[1,1,1,1]]
            masks = tf.constant(masks,dtype=tf.uint8)
            masks = tf.reshape(masks,[4,2,2])
            area = odm.mask_area(masks)
            sess.run(tf.global_variables_initializer())
            area = sess.run(area)
            self.assertAllClose(area,[2. ,  1.,    0.,4.],atol=1e-3)




if __name__ == "__main__":
    np.random.seed(int(time.time()))
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s %(funcName)s:%(message)s',
                        datefmt="%H:%M:%S")
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.test.main()
