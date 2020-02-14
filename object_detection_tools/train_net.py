#coding=utf-8
import object_detection2.config.config as config
from object_detection2.engine.train_loop import SimpleTrainer
from object_detection2.engine.defaults import default_argument_parser
from object_detection2.data.dataloader import *
from object_detection2.data.datasets.build import DATASETS_REGISTRY
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
CHECK_POINT_FILE_NAME = "data.ckpt"


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = config.get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.log_dir = args.log_dir
    cfg.ckpt_dir = args.ckpt_dir
    return cfg


def main(_):
    args = default_argument_parser().parse_args()

    cfg = setup(args)

    if args.eval_only:
        '''model = SimpleTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res'''
        pass

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    data_loader = DataLoader(cfg=cfg)
    data_args = DATASETS_REGISTRY[cfg.DATASETS.TRAIN]
    data,num_classes = data_loader.load_data(*data_args)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.SSD.NUM_CLASSES = num_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    cfg.DATASETS.NUM_CLASSES = num_classes
    cfg.freeze()
    config.set_global_cfg(cfg)

    model = SimpleTrainer.build_model(cfg,is_training=True)

    trainer = SimpleTrainer(cfg,data=data,model=model)
    trainer.resume_or_load()
    return trainer.train()



if __name__ == "__main__":
    tf.app.run()
