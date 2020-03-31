#coding=utf-8
import object_detection2.config.config as config
from object_detection2.engine.train_loop import SimpleTrainer
from object_detection2.engine.defaults import default_argument_parser
from object_detection2.data.dataloader import *
from object_detection2.data.datasets.build import DATASETS_REGISTRY
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
CHECK_POINT_FILE_NAME = "data.ckpt"


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


def main(_):
    is_training = False
    args = default_argument_parser().parse_args()

    cfg = setup(args)
    data_loader = DataLoader(cfg=cfg,is_training=is_training)
    data_args = DATASETS_REGISTRY[cfg.DATASETS.TEST]
    data,num_classes = data_loader.load_data(*data_args,batch_size=1,is_training=False)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.SSD.NUM_CLASSES = num_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    cfg.DATASETS.NUM_CLASSES = num_classes
    cfg.freeze()
    config.set_global_cfg(cfg)

    model = SimpleTrainer.build_model(cfg,is_training=is_training)
    trainer = SimpleTrainer(cfg,data=data,model=model)
    trainer.resume_or_load()
    res = trainer.test(cfg, model)
    return res

if __name__ == "__main__":
    tf.app.run()
