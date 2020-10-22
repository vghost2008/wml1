#coding=utf-8
import tensorflow as tf
import wml_utils as wmlu
import object_detection2.config as config
import argparse
import os

def default_argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="ObjectDetection2 Training")
    #parser.add_argument("--config-file", default="Base-RetinaNet.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="RetinaNet-anchor.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="RetinaNet.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="EfficientDet-DR.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="Mask-RCNN-FPN-sephv2.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="Mask-RCNN-FPN-3-2.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="RetinaNetBBD.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--config-file", default="Mask-RCNN-FPN-gnn.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="Mask-RCNN-FPN-box-free.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--research-file", default="research.txt", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--log_dir", default=wmlu.home_dir("ai/tmp3/object_detection2_log"),type=str,help="path to log dir")
    parser.add_argument("--ckpt_dir", default=wmlu.home_dir("ai/tmp3/object_detection2"),type=str,help="path to ckpt dir")
    parser.add_argument("--test_data_dir", default="/2_data/wj/mldata/coco/val2017",type=str,help="path to test data dir")
    parser.add_argument("--save_data_dir", default="/2_data/wj/mldata/coco/coco_results",type=str,help="path to save data dir")
    '''
    begin training datetime:
    format: YY-MM-dd HH:MM:SS如20-06-22 15:12:00
    '''
    parser.add_argument("--runtime", default="", type=str, help="datetime to begin tarning.")
    parser.add_argument("--gpus", nargs='+',type=int, help="gpus for training.")
    parser.add_argument("--restore", type=str, help="restore option.",default="auto")  #auto, ckpt,finetune, none
    return parser

def get_config_file(name:str):
    CONFIG_DIR = "/home/vghost/ai/work/wml/object_detection2/default_configs/"
    COCOCONFIG_DIR = "/home/vghost/ai/work/wml/object_detection2/default_configs/coco/"
    MODCONFIG_DIR = "/home/vghost/ai/work/wml/object_detection2/default_configs/mnistod/"
    search_dirs = [COCOCONFIG_DIR,MODCONFIG_DIR,CONFIG_DIR]
    if os.path.exists(name):
        return name
    if not name.endswith(".yaml"):
        name = name+".yaml"

    for dir in search_dirs:
        path = os.path.join(dir,name)
        if os.path.exists(path):
            return path

    return name
