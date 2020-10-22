#coding=utf-8
import object_detection2.config.config as config
from object_detection2.engine.train_loop import SimpleTrainer
from object_detection2.data.dataloader import DEFAULT_CATEGORY_INDEX
from object_detection2.data.dataloader import *
import wml_tfutils as wmlt
import numpy as np
import img_utils as wmli
import glob
import tensorflow as tf
import os
import sys
from object_detection2.datadef import *
from object_detection2.standard_names import *

import wml_utils as wmlu

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
slim = tf.contrib.slim

class PredictModel(object):
    def __init__(self,input_shape=[1,None,None,3],cfg=None,category_index=None,is_remove_batch=True):

        self.input_imgs = tf.placeholder(tf.uint8,input_shape)        
        is_training = False
        model = SimpleTrainer.build_model(cfg, is_training=is_training)
        self.cfg = cfg
        imgs = tf.cast(self.input_imgs,tf.float32)
        self.trainer = SimpleTrainer(cfg, data=imgs, model=model,inference=True)
        self.trainer.category_index = DEFAULT_CATEGORY_INDEX if category_index is not None else category_index
        self.log_step = 100
        self.have_mask = cfg.MODEL.MASK_ON
        self.timer = wmlu.AvgTimeThis(skip_nr=5)
        if RD_MASKS in self.trainer.res_data:
            self.trainer.draw_on_image()
            self.trainer.add_full_size_mask()
        if is_remove_batch:
            self.remove_batch()
        print("Use default dev.")
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.merged = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer = tf.summary.FileWriter(self.trainer.log_dir, self.sess.graph)
        self.step = 0

    def restoreVariables(self):
        self.trainer.resume_or_load(sess=self.sess)

    def predictImages(self,imgs):
        print(imgs.shape)
        feed_dict = {self.input_imgs:imgs}
        if self.step % self.log_step == 0:
            res_data = dict(self.trainer.res_data_for_eval)
            summary_dict = {"summary": self.merged}
            res_data.update(summary_dict)
            res_data = self.sess.run(res_data,feed_dict=feed_dict)
            self.summary_writer.add_summary(res_data['summary'], self.step)
            sys.stdout.flush()
        else:
            res_data = self.trainer.res_data_for_eval
            with self.timer:
                res_data = self.sess.run(res_data,feed_dict=feed_dict)
        self.res_data = res_data
        self.step += 1
        
        return self.res_data

    def remove_batch(self):
        assert self.trainer.res_data_for_eval[RD_BOXES].get_shape().as_list()[0]==1, "error batch size"
        len = self.trainer.res_data_for_eval[RD_LENGTH][0]
        data = self.trainer.res_data_for_eval
        if self.cfg.MODEL.MASK_ON:
            self.trainer.res_data_for_eval[RD_MASKS] = data[RD_MASKS][0][:len]
        self.trainer.res_data_for_eval[RD_BOXES] = data[RD_BOXES][0][:len]
        self.trainer.res_data_for_eval[RD_LABELS] = data[RD_LABELS][0][:len]
        self.trainer.res_data_for_eval[RD_PROBABILITY] = data[RD_PROBABILITY][0][:len]
        if RD_FULL_SIZE_MASKS in self.trainer.res_data_for_eval:
            self.trainer.res_data_for_eval[RD_FULL_SIZE_MASKS] = data[RD_FULL_SIZE_MASKS][0][:len]
        if RD_RESULT_IMAGE in self.trainer.res_data_for_eval:
            self.trainer.res_data_for_eval[RD_RESULT_IMAGE] = data[RD_RESULT_IMAGE][0]
