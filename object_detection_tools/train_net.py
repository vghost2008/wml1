#coding=utf-8
import object_detection2.config.config as config
from object_detection2.engine.train_loop import SimpleTrainer
from object_detection2.engine.defaults import default_argument_parser
from object_detection2.data.dataloader import *
from object_detection2.data.datasets.build import DATASETS_REGISTRY
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

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


def main(_):
    args = default_argument_parser().parse_args()

    cfg = setup(args)

    is_training = True
    data_loader = DataLoader(cfg=cfg,is_training=is_training)
    #'''
    data_loader.trans_on_single_img = [trans.MaskNHW2HWN(),
                                trans.ResizeToFixedSize([256,256]),
                                trans.MaskHWN2NHW(),
                                trans.AddBoxLens(),
                                trans.UpdateHeightWidth(),
                                ]
    data_loader.trans_on_batch_img = [trans.FixDataInfo()]
    #'''
    data_args = DATASETS_REGISTRY[cfg.DATASETS.TRAIN]
    with tf.device(":/cpu:0"):
        data,num_classes = data_loader.load_data(*data_args)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.SSD.NUM_CLASSES = num_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    cfg.DATASETS.NUM_CLASSES = num_classes
    cfg.freeze()
    config.set_global_cfg(cfg)

    model = SimpleTrainer.build_model(cfg,is_training=is_training)

    trainer = SimpleTrainer(cfg,data=data,model=model)
    if len(cfg.MODEL.WEIGHTS) > 3:
        kwargs = {}
        if cfg.MODEL.ONLY_SCOPE != "":
            kwargs["only_scope"] = cfg.MODEL.ONLY_SCOPE
    else:
        kwargs = {'extend_vars': trainer.global_step}
    trainer.resume_or_load(**kwargs)
    return trainer.train()

if __name__ == "__main__":
    tf.app.run()
