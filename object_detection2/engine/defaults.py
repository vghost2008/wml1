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
    parser = argparse.ArgumentParser(description="Arguments")
    #parser.add_argument("--config-file", default="cascade_mask_FPN_M", metavar="FILE", help="path to config file")
    parser.add_argument("--config-file", default="FCOS_M.yaml", metavar="FILE", help="path to config file")
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
    parser.add_argument("--log_dir", default=wmlu.home_dir("ai/tmp/object_detection2_log"),type=str,help="path to log dir")
    parser.add_argument("--ckpt_dir", default=wmlu.home_dir("ai/tmp/object_detection2"),type=str,help="path to ckpt dir")
    parser.add_argument("--test_data_dir", default="/2_data/wj/mldata/coco/val2017",type=str,help="path to test data dir")
    parser.add_argument("--save_data_dir", default="/2_data/wj/mldata/coco/coco_results",type=str,help="path to save data dir")
    '''
    begin training datetime:
    format: YY-MM-dd HH:MM:SS如20-06-22 15:12:00
    '''
    parser.add_argument("--runtime", default="", type=str, help="datetime to begin tarning.")
    parser.add_argument("--gpus", nargs='+',type=int, help="gpus for training.")
    parser.add_argument("--restore", type=str, help="restore option.",default="auto")  #auto, ckpt,finetune, none
    parser.add_argument("--confthre", type=float, help="confidence threshold for predict on images.",default=None)  #auto, ckpt,finetune, none
    parser.add_argument("--save_pb_path", type=str, help="save pb path.",default="model.pb")
    return parser

def get_config_file(name:str):
    CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),"default_configs")
    COCOCONFIG_DIR = os.path.join(CONFIG_DIR,"coco")
    MODCONFIG_DIR = os.path.join(CONFIG_DIR,"mnistod")
    MODGEOCONFIG_DIR = os.path.join(CONFIG_DIR,"mnistodgeo")
    RESEARCH_DIR = os.path.join(CONFIG_DIR,"research")
    SEMANTIC_DIR = os.path.join(CONFIG_DIR,"semantic")
    KEYPOINTS = os.path.join(CONFIG_DIR,"keypoints")
    MOT = os.path.join(CONFIG_DIR,"MOT")
    search_dirs = [SEMANTIC_DIR,COCOCONFIG_DIR,MODCONFIG_DIR,CONFIG_DIR,RESEARCH_DIR,MODGEOCONFIG_DIR,KEYPOINTS,MOT]
    if os.path.exists(name):
        return name
    if not name.endswith(".yaml"):
        name = name+".yaml"

    for dir in search_dirs:
        path = os.path.join(dir,name)
        if os.path.exists(path):
            return path

    ALL_FILES = wmlu.recurse_get_filepath_in_dir(CONFIG_DIR,suffix="yaml")

    for file in ALL_FILES:
        if os.path.basename(file)==name:
            return file

    return name
