#!/home/wj/anaconda3_rl/bin/python
#coding=utf-8
from predictmodel import *
from object_detection2.engine.defaults import default_argument_parser, get_config_file
import os
from object_detection2.data.datasets.build import DATASETS_REGISTRY

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
    if len(cfg.log_dir)==0:
        cfg.log_dir = args.log_dir
    if len(cfg.ckpt_dir)==0:
        cfg.ckpt_dir = args.ckpt_dir
    return cfg

def add_version_info():
    t = time.localtime(time.time())
    t_str = time.strftime("%Y-%m-%d %H:%M:%S", t)
    VERSION_INFO = "12-0_" + " V21.1" + "_" + t_str + "; "
    get_info = tf.placeholder(tf.int32, shape=(), name="get_info")
    info = tf.case([(tf.equal(get_info, 0), lambda: tf.constant(VERSION_INFO))], default=lambda: tf.constant("N.A.; "))
    info = tf.identity(info, "info")

def main(_):
    is_training = False
    args = default_argument_parser().parse_args()

    cfg = setup(args)
    data_loader = DataLoader(cfg=cfg, is_training=is_training)
    data_args = DATASETS_REGISTRY[cfg.DATASETS.TEST]
    data, num_classes = data_loader.load_data(*data_args, batch_size=1, is_training=False)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SSD.NUM_CLASSES = num_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    cfg.MODEL.CENTERNET.NUM_CLASSES = num_classes
    cfg.MODEL.YOLACT.NUM_CLASSES = num_classes
    cfg.MODEL.FCOS.NUM_CLASSES = num_classes
    cfg.DATASETS.NUM_CLASSES = num_classes
    cfg.freeze()
    config.set_global_cfg(cfg)

    model = PredictModel(cfg=cfg,is_remove_batch=True)
    model.restoreVariables()

    model.savePBFile(args.save_pb_path)

if __name__ == "__main__":
    tf.app.run()

'''
python object_detection_tools/tf2pb.py  --gpus 3  --config-file gds1v2 --save_pb_path model.pb
'''
