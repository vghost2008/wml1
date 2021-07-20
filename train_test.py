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
import tempfile
from wnn import *
from tensorflow.contrib.opt import MovingAverageOptimizer
CHECK_POINT_FILE_NAME = "data.ckpt"


class WMLTest(tf.test.TestCase):
    def testMultiTrain(self):
        global_step = tf.train.get_or_create_global_step()
        x = tf.Variable(50, dtype=tf.float32,name="x")
        y = tf.Variable(50, dtype=tf.float32,name="y")
        tower_grads = []
        opt = get_optimizer(global_step, optimizer="GD",learning_rate=5.0,
                            batch_size=1,num_epochs_per_decay=100,example_size=1,
                            learn_rate_decay_factor=0.2,min_learn_rate=1e-5)
        for i in range(2):
            with tf.device("/cpu:{}".format(0)):
                with tf.name_scope("cpu_{}".format(i)):
                    loss = tf.pow(x - 10.0, 2) + 9.0 + tf.pow(y - 5., 2)
                    loss = tf.reduce_sum(loss)
                    tf.losses.add_loss(loss)

                    grads, _, _ = get_train_opv3(optimizer=opt, loss=loss)
                    tower_grads.append(grads)

        avg_grads = average_grads(tower_grads, clip_norm=1)
        opt0 = apply_gradientsv3(avg_grads, global_step, opt)
        g_x = avg_grads[0][0]
        g_y = avg_grads[1][0]
        opt1 = get_batch_norm_ops()
        train_op = tf.group(opt0, opt1)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for i in range(300):
            r_x, r_y, rgx,rgy,l,_ = sess.run([x, y, g_x,g_y,loss,train_op])
            #print(r_x, r_y,rgx,rgy,l)
        self.assertAllClose([r_x,r_y],[10,5],atol=1e-4)
    def testMovingAverageOptimizer(self):
        with self.test_session() as sess:
            global_step = tf.train.get_or_create_global_step()
            x = tf.Variable(50, dtype=tf.float32,name="x")
            y = tf.Variable(50, dtype=tf.float32,name="y")
            tower_grads = []
            opt = get_optimizer(global_step, optimizer="GD",learning_rate=1.0,
                                batch_size=1,num_epochs_per_decay=100,example_size=1,
                                learn_rate_decay_factor=0.2,min_learn_rate=1e-5)
            opt = MovingAverageOptimizer(opt,average_decay=0.8)
            for i in range(2):
                with tf.device("/cpu:{}".format(0)):
                    with tf.name_scope("cpu_{}".format(i)):
                        loss = tf.pow(x - 10.0, 2) + 9.0 + tf.pow(y - 5., 2)
                        loss = tf.reduce_sum(loss)
                        tf.losses.add_loss(loss)

                        grads, _, _ = get_train_opv3(optimizer=opt, loss=loss)
                        tower_grads.append(grads)

            avg_grads = average_grads(tower_grads, clip_norm=1)
            opt0 = apply_gradientsv3(avg_grads, global_step, opt)
            g_x = avg_grads[0][0]
            g_y = avg_grads[1][0]
            opt1 = get_batch_norm_ops()
            train_op = tf.group(opt0, opt1)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            for i in range(60):
                r_x, r_y, rgx,rgy,l,_ = sess.run([x, y, g_x,g_y,loss,train_op])
                print(f"step {i}: ",r_x, r_y,rgx,rgy,l)
            saver = opt.swapping_saver()
            check_point_dir = tempfile.gettempdir()
            ckpt_path = os.path.join(check_point_dir, CHECK_POINT_FILE_NAME)
            saver.save(sess,ckpt_path, global_step=1)

    def testMovingAverageOptimizer1(self):
        with self.test_session() as sess:
            check_point_dir = tempfile.gettempdir()
            ckpt_path = os.path.join(check_point_dir, CHECK_POINT_FILE_NAME)
            x = tf.Variable(50, dtype=tf.float32,name="x")
            y = tf.Variable(50, dtype=tf.float32,name="y")
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver([x,y])
            saver.restore(sess,ckpt_path+"-1")
            wmlt.check_value_in_ckp(sess,"x")
            wmlt.check_value_in_ckp(sess,"y")



if __name__ == "__main__":
    np.random.seed(int(time.time()))
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s %(funcName)s:%(message)s',
                        datefmt="%H:%M:%S")
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.test.main()
