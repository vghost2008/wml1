#coding=utf-8
import object_detection2.config.config as config
from object_detection2.engine.train_loop import SimpleTrainer
from object_detection2.engine.defaults import default_argument_parser
from object_detection2.data.dataloader import *
from object_detection2.data.datasets.build import DATASETS_REGISTRY
import tensorflow as tf
import os
import sys
import datetime
import time

gpus = [0,1,2]
gpus = [2,3,4]
#gpus = [1,5,6]
#gpus = [3,6,7]
gpus = [3]

slim = tf.contrib.slim
def setup(args):
    global gpus
    """
    Create configs and perform basic setups.
    """
    cfg = config.get_cfg()
    if args.gpus is not None:
        gpus = args.gpus
        
    gpus_str = ""
    for g in gpus:
        gpus_str += str(g) + ","
    gpus_str = gpus_str[:-1]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_str

    config_path = config.get_config_file(args.config_file)
    print(f"Config file {args.config_file}, gpus={os.environ['CUDA_VISIBLE_DEVICES']}")
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(args.opts)
    cfg.log_dir = args.log_dir
    cfg.ckpt_dir = args.ckpt_dir
    return cfg

def main(_):
    args = default_argument_parser().parse_args()

    cfg = setup(args)

    is_training = True
    data_loader = DataLoader(cfg=cfg,is_training=is_training)
    data_args = DATASETS_REGISTRY[cfg.DATASETS.TRAIN]
    with tf.device(":/cpu:0"):
        data,num_classes = data_loader.load_data(*data_args)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.SSD.NUM_CLASSES = num_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    cfg.MODEL.CENTERNET.NUM_CLASSES = num_classes
    cfg.MODEL.YOLACT.NUM_CLASSES = num_classes
    cfg.MODEL.FCOS.NUM_CLASSES = num_classes
    cfg.DATASETS.NUM_CLASSES = num_classes
    cfg.DATASETS.NUM_CLASSES = num_classes

    cfg.freeze()
    config.set_global_cfg(cfg)

    '''
    用于指定在将来的某个时间点执行任务
    '''
    if len(args.runtime) > 12:
        target_datetime = datetime.datetime.strptime(args.runtime, "%Y-%m-%d %H:%M:%S")
        while True:
            wait_time = (target_datetime - datetime.datetime.now()).total_seconds() / 3600.0
            sys.stdout.write(
                f"\rRumtime is {target_datetime}, current datetime is {datetime.datetime.now()}, need to wait for {wait_time:.2f}h", )
            sys.stdout.flush()

            if datetime.datetime.now() >= target_datetime:
                break
            else:
                time.sleep(30)
                
    print("")
    model = SimpleTrainer.build_model(cfg,is_training=is_training)

    trainer = SimpleTrainer(cfg,data=data,model=model,gpus=gpus,debug_tf=False)
    kwargs = {}
    if args.restore == "auto":
        if len(cfg.MODEL.WEIGHTS) > 3:
            if cfg.MODEL.ONLY_SCOPE != "":
                kwargs["only_scope"] = cfg.MODEL.ONLY_SCOPE

            def func(v):
                name = v.name[:-2]
                if "BatchNorm" in name:
                    name = name.replace("BatchNorm","tpu_batch_normalization")
                return name
            #kwargs['value_key'] = func
        else:
            kwargs = {'extend_vars': trainer.global_step}
    elif args.restore == "ckpt":
        kwargs = {'extend_vars': trainer.global_step}
        pass
    elif args.restore == "finetune":
        kwargs = {'extend_vars': trainer.global_step}
        pass
    elif args.restore == "none":
        pass
    else:
        raise NotImplementedError("Error")
    trainer.resume_or_load(**kwargs,option=args.restore)
    return trainer.train()

if __name__ == "__main__":
    tf.app.run()
