# coding=utf-8
import object_detection2.config.config as config
from object_detection2.standard_names import *
from object_detection2.engine.defaults import default_argument_parser, get_config_file
from object_detection2.data.dataloader import *
from object_detection2.data.datasets.build import DATASETS_REGISTRY
import tensorflow as tf
import os
from object_detection_tools.predictmodel import PredictModel
import wml_utils as wmlu
import img_utils as wmli
import object_detection_tools.visualization as odv
import numpy as np
from object_detection2.data.datasets.buildin import coco_category_index
from iotoolkit.coco_toolkit import COCOData
from iotoolkit.pascal_voc_toolkit import PascalVOCData
from object_detection2.metrics.toolkit import *

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

def eval_dataset(data_dir):
    '''data = COCOData()
    data.read_data(wmlu.home_dir("ai/mldata/coco/annotations/instances_val2014.json"),
                   image_dir=wmlu.home_dir("ai/mldata/coco/val2014"))'''
    def label_text2id(x):
        data =  {"scratch":1}
        return data[x]
    data = PascalVOCData(label_text2id=label_text2id)
    data.read_data(data_dir,img_suffix=".bmp;;.jpg;;.jpeg")

    return data.get_items()

def text_fn(label,probability):
    return f"{label}:{probability:.2f}"
    
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

    save_path = args.save_data_dir

    wmlu.create_empty_dir(save_path,remove_if_exists=True)
    metrics = COCOEvaluation(num_classes=90)

    items = eval_dataset(args.test_data_dir)
    
    for data in items:
        full_path, shape, gt_labels, category_names, gt_boxes, binary_masks, area, is_crowd, num_annotations_skipped = data
        img = wmli.imread(full_path)
        imgs = np.expand_dims(img,axis=0)
        res = model.predictImages(imgs)

        if RD_MASKS in res:
            r_img = odv.draw_bboxes_and_mask(img,res[RD_LABELS],res[RD_PROBABILITY],res[RD_BOXES],
                                     res[RD_MASKS],
                                             show_text=True)
        else:
            r_img = odv.bboxes_draw_on_imgv2(img,res[RD_LABELS],res[RD_PROBABILITY],res[RD_BOXES],
                                             text_fn=text_fn,
                                             show_text=True)
        kwargs = {}
        kwargs['gtboxes'] = gt_boxes
        kwargs['gtlabels'] =gt_labels 
        kwargs['boxes'] = res[RD_BOXES]
        kwargs['labels'] = res[RD_LABELS] 
        kwargs['probability'] = res[RD_PROBABILITY] 
        kwargs['img_size'] = shape
        metrics(**kwargs)
        
        if model.step%100 == 0:
            metrics.show()
        
        name = wmlu.base_name(full_path)
        img_save_path = os.path.join(save_path,name+".png")
        wmli.imwrite(img_save_path,r_img)
        
    metrics.show()

if __name__ == "__main__":
    tf.app.run()

'''
python object_detection_tools/eval_on_images.py --test_data_dir ~/ai/mldata1/GDS1Crack/val/ng/ --gpus 3 --config-file gds1 --save_data_dir ~/ai/mldata1/GDS1Crack/tmp/gds1_output
0.114|0.171
'''