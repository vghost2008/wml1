# coding=utf-8
import object_detection2.config.config as config
from object_detection2.engine.train_loop import SimpleTrainer
from object_detection2.engine.defaults import default_argument_parser, get_config_file
from object_detection2.data.dataloader import *
from object_detection2.data.datasets.build import DATASETS_REGISTRY
from object_detection_tools.predictmodel import PredictModel
from object_detection2.mot_toolkit.fair_mot_tracker.multitracker import JDETracker
from object_detection2.mot_toolkit.fair_mot_tracker.multitracker_cpp import CPPTracker
from object_detection2.standard_names import *
from object_detection2.mot_toolkit.build import build_mot
import tensorflow as tf
import os
import os.path as osp
import wml_utils as wmlu
import img_utils as wmli
import cv2


slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
CHECK_POINT_FILE_NAME = "data.ckpt"

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = config.get_cfg()
    if args.gpus is not None:
        gpus = args.gpus
    else:
        gpus = []

    gpus_str = ""
    for g in gpus:
        gpus_str += str(g) + ","
    gpus_str = gpus_str[:-1]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_str
    print(f"Config file {args.config_file}")
    config_path = get_config_file(args.config_file)
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(args.opts)
    cfg.log_dir = args.log_dir
    cfg.ckpt_dir = args.ckpt_dir
    return cfg


def main(_):
    is_training = False
    args = default_argument_parser().parse_args()

    cfg = setup(args)
    data_args = DATASETS_REGISTRY[cfg.DATASETS.TEST]
    cfg.MODEL.NUM_CLASSES = data_args[2]
    data_loader = DataLoader(cfg=cfg, is_training=is_training)
    data, num_classes = data_loader.load_data(*data_args, batch_size=1, is_training=False)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SSD.NUM_CLASSES = num_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    cfg.MODEL.CENTERNET.NUM_CLASSES = num_classes
    cfg.MODEL.DEEPLAB.NUM_CLASSES = num_classes
    cfg.MODEL.YOLACT.NUM_CLASSES = num_classes
    cfg.MODEL.FCOS.NUM_CLASSES = num_classes
    cfg.MODEL.NUM_CLASSES = num_classes
    cfg.DATASETS.NUM_CLASSES = num_classes
    cfg.freeze()
    config.set_global_cfg(cfg)

    #model = PredictModel(cfg=cfg,is_remove_batch=False,input_shape=[1,256,480,3],input_name="input",input_dtype=tf.float32)
    model = PredictModel(cfg=cfg,is_remove_batch=False,input_shape=[1,None,None,3],input_name="input",input_dtype=tf.float32)
    #model = PredictModel(cfg=cfg,is_remove_batch=False,input_shape=[1,None,None,3],input_name="input",input_dtype=tf.float32)
    rename_dict = {RD_BOXES:"bboxes",RD_PROBABILITY:"probs",RD_ID:"id"}
    model.remove_batch_and_rename(rename_dict)
    model.restoreVariables()
    names = ["shared_head/heat_ct/Conv_1/BiasAdd",
            "shared_head/ct_regr/Conv_1/BiasAdd",
            "shared_head/hw_regr/Conv_1/BiasAdd",
            "shared_head/l2_normalize"]

    #model.savePBFile("/home/wj/0day/test.pb",["bboxes","probs","id"])
    #model.savePBFile("/home/wj/0day/mot_large.pb",names)
    model.savePBFile("/home/wj/0day/motv3.pb",names)
    #return
    tracker = build_mot(cfg,model)
    #model.savePBFile("/home/wj/0day/test.pb",names)
    #return
    #tracker = JDETracker(model)
    #tracker = CPPTracker(model)

    path = '/home/wj/ai/mldata/MOT/MOT20/test/MOT20-04'
    path = '/home/wj/ai/mldata/MOT/MOT20/test_1img'
    path = '/home/wj/ai/mldata/MOT/MOT15/test2/TUD-Crossing/img1'
    path = '/home/wj/ai/mldata/pose3d/basketball2.mp4'
    path = '/home/wj/ai/mldata/pose3d/tennis1.mp4'
    #files = wmlu.recurse_get_filepath_in_dir(args.test_data_dir,suffix=".jpg")
    files = wmlu.recurse_get_filepath_in_dir(path,suffix=".jpg")
    #save_path = args.save_data_dir
    save_dir = '/home/wj/ai/mldata/MOT/output2'
    save_path = osp.join(save_dir,osp.basename(path))
    wmlu.create_empty_dir(save_dir,remove_if_exists=True,yes_to_all=True)
    writer = wmli.VideoWriter(save_path)
    reader = wmli.VideoReader(path)

    for img in reader:
        img = wmli.resize_width(img,960)
        objs = tracker.update(img)
        img = tracker.draw_tracks(img,objs)
        writer.write(img)
    writer.release()

if __name__ == "__main__":
    tf.app.run()
