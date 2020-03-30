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
    parser = argparse.ArgumentParser(description="Detectron2 Training")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-RPN.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-RPN-FPN.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-RPN-FPN_r.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-RCNN-C4.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-RCNN-FPN.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-RCNN-FPN_r.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-Mask-RCNN-FPN-C4.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-Mask-RCNN-FPN-C4_3x.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/cascade_mask_FPN_1x.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/cascade_mask_FPN_3x.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-RetinaNet.yaml", metavar="FILE", help="path to config file")
    #parser.add_argument("--config-file", default="/home/vghost/ai/work/wml/object_detection2/default_configs/Base-SSD.yaml", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    #parser.add_argument("--eval-only", type=bool,default=True, help="perform evaluation only")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
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
