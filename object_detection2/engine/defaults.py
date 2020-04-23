#coding=utf-8
import tensorflow as tf
import wml_utils as wmlu
import object_detection2.config as config
import argparse

def default_argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Returns:
        argparse.ArgumentParser:
    """
    CONFIG_DIR = "/home/vghost/ai/work/wml/object_detection2/default_configs/"
    COCOCONFIG_DIR = "/home/vghost/ai/work/wml/object_detection2/default_configs/coco/"
    parser = argparse.ArgumentParser(description="ObjectDetection2 Training")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-RPN.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-RPN-FPN.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-RPN-FPN_r.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-RCNN-C4.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-RCNN-FPN.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-RCNN-FPN_r.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-Mask-RCNN-FPN-C4.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-Mask-RCNN-FPN-C4_3x.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-Mask-RCNN-FPN-GIOU-C4.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-Mask-RCNN-FPN-GIOU-C4-2.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-Mask-RCNN-FPN-RETINARPN-GIOU-C4.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-Mask-RCNN-FPN-RETINARPN-C4.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/cascade_mask_FPN_1x.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/cascade_mask_FPN_3x.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-SSD.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-YOLACT.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default=CONFIG_DIR+"Base-RetinaNet.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default=COCOCONFIG_DIR+"RetinaNet_4.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--config-file", default=COCOCONFIG_DIR+"EfficientDet-D0.yaml", metavar="FILE", help="path to config file")
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
    parser.add_argument("--log_dir", default=wmlu.home_dir("ai/tmp/"),type=str,help="path to log dir")
    parser.add_argument("--ckpt_dir", default=wmlu.home_dir("ai/tmp/"),type=str,help="path to ckpt dir")
    parser.add_argument("--test_data_dir", default=wmlu.home_dir("ai/tmp/"),type=str,help="path to test data dir")
    parser.add_argument("--save_data_dir", default=wmlu.home_dir("ai/tmp/"),type=str,help="path to save data dir")
    return parser
