#coding=utf-8
import object_detection2.config.config as config
from object_detection2.engine.train_loop import SimpleTrainer
from object_detection2.engine.defaults import default_argument_parser
from object_detection2.data.dataloader import *
from object_detection2.data.datasets.build import DATASETS_REGISTRY
import wml_tfutils as wmlt
import numpy as np
import img_utils as wmli
import glob
import tensorflow as tf
import os
import sys
from object_detection2.datadef import *
from object_detection2.standard_names import *
import iotoolkit.pascal_voc_toolkit as pvt

import wml_utils as wmlu
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
slim = tf.contrib.slim

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = config.get_cfg()
    print(f"Config file {args.config_file}")
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.log_dir = args.log_dir
    cfg.ckpt_dir = args.ckpt_dir
    return cfg

class PredictModel(object):
    def __init__(self,input_shape=[1,None,None,3]):

        self.input_imgs = tf.placeholder(tf.uint8,input_shape)        
        is_training = False
        args = default_argument_parser().parse_args()
        cfg = setup(args)
        num_classes = cfg.DATASETS.NUM_CLASSES #object detection2's num classes dosen't include background
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
        cfg.MODEL.SSD.NUM_CLASSES = num_classes
        cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
        cfg.DATASETS.NUM_CLASSES = num_classes
        cfg.freeze()
        config.set_global_cfg(cfg)
        model = SimpleTrainer.build_model(cfg, is_training=is_training)
        self.cfg = cfg
        self.trainer = SimpleTrainer(cfg, data=self.input_imgs, model=model,inference=True)
        self.log_step = 100
        #self.have_mask = cfg.MODEL.MASK_ON
        self.have_mask = False
        self.timer = wmlu.AvgTimeThis(skip_nr=5)

    def restoreVariables(self,ckpt_path):
        self.trainer.resume_or_load(sess=self.sess)


    def buildNet(self):
        print("Use default dev.")
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.merged = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer = tf.summary.FileWriter(self.trainer.log_dir, self.sess.graph)

    def predictImages(self,imgs):
        feed_dict = {self.input_imgs:imgs}
        print(imgs.shape,imgs.dtype)
        if self.step % self.log_step == 0:
            res_data = self.trainer.res_data_for_eval
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

    def remove_batch(self):
        assert self.trainer.res_data_for_eval[RD_BOXES].get_shape().as_list()[0]==1, "error batch size"
        len = self.trainer.res_data_for_eval[RD_LENGTH][0]
        data = self.trainer.res_data_for_eval
        if self.cfg.MODEL.MASK_ON:
            self.trainer.res_data_for_eval[RD_MASKS] = data[RD_MASKS][0][:len]
        self.trainer.res_data_for_eval[RD_BOXES] = data[RD_BOXES][0][:len]
        self.trainer.res_data_for_eval[RD_LABELS] = data[RD_LABELS][0][:len]
        self.trainer.res_data_for_eval[RD_PROBABILITY] = data[RD_PROBABILITY][0][:len]

def main():
    args = default_argument_parser().parse_args()
    test_dir = args.test_data_dir
    save_dir = args.save_data_dir
    wmlu.create_empty_dir(save_dir,remove_if_exists=True)
    files =  glob.glob(os.path.join(test_dir,"*.jpg"))
    m = PredictModel()
    m.remove_batch()
    for file in files:
        img = wmli.imread(file)
        img = np.expand_dims(img,axis=0)
        m.predictImages(img)
        save_path = os.path.join(save_dir,os.path.basename(file))
        xml_path = wmlu.change_suffix(save_path,"xml")
        shutil.copy(file,save_path)
        pvt.writeVOCXml(xml_path,m.res_data[RD_BOXES],m.res_data[RD_LABELS])
