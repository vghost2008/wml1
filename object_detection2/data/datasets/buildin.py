# -*- coding: utf-8 -*-
import os
from .build import DATASETS_REGISTRY
from iotoolkit.pascal_voc_tf_decodev2 import get_data as voc_get_data
from iotoolkit.coco_tf_decodev2 import get_data as coco_get_data
from iotoolkit.coco_tf_kp_deccode import get_data as coco_kp_get_data
from iotoolkit.coco_toolkit import ID_TO_TEXT as coco_id_to_text
from iotoolkit.coco_toolkit import COMPRESSED_ID_TO_TEXT as coco_compressed_id_to_text
from iotoolkit.pascal_voc_data import ID_TO_TEXT as pascal_voc_id_to_text
from iotoolkit.mot_tf_decode import get_data as mot_get_data

# ==== Predefined datasets and splits for COCO ==========
dataset_root_path = "/home/wj/ai/mldata"

default_category_index = None
coco_category_index = {}
pascal_voc_category_index = {}
mod_category_index = {1:"rectangle",2:"triangle",3:"ellipse"}
modgeo_category_index = {
    1:"NSNN",
    2:"SNN",
    3:"SUN",
    4:"SNU",
    5:"SUU",
    6:"BNN",
    7:"NBNN",
    8:"BUN",
    9:"BNU",
    10:"BUU",
}
for k,v in coco_id_to_text.items():
    coco_category_index[k] = v['name']
for k,v in pascal_voc_id_to_text.items():
    pascal_voc_category_index[k] = v


def register_all_pascal_voc(root="datasets"):
    #名字，tfrecord文件路径,解码函数，num_classes(不包含背景)
    SPLITS = [
        ("voc_2012_train", os.path.join(dataset_root_path,"VOC2012_tfdata"),voc_get_data,20)
    ]
    for x in SPLITS:
        name = x[0]
        args = x[1:]
        DATASETS_REGISTRY.register(name,args)
def register_all_coco(root="datasets"):
    #名字，tfrecord文件路径,解码函数，num_classes(不包含背景)
    SPLITS = [
        ("coco_2017_train", os.path.join(dataset_root_path,"coco/tfdata_2017_train"),coco_get_data,80,coco_compressed_id_to_text),
        ("coco_2017_eval", os.path.join(dataset_root_path,"coco/tfdata_2017_val"),coco_get_data,80,coco_compressed_id_to_text),
        ("coco_2017_kp_train", os.path.join(dataset_root_path,"coco/tfdata_2017_kp_train"),coco_kp_get_data,80,coco_compressed_id_to_text),
        #("coco_2017_kp_train", os.path.join(dataset_root_path,"coco/tfdata_2017_kp_val"),coco_kp_get_data,80,coco_compressed_id_to_text),
        ("coco_2017_kp_eval", os.path.join(dataset_root_path,"coco/tfdata_2017_kp_val"),coco_kp_get_data,80,coco_compressed_id_to_text),
        ("coco_2014_train", os.path.join(dataset_root_path,"coco/tfdata1"),coco_get_data,90,coco_category_index),
        ("coco_2014_eval", os.path.join(dataset_root_path,"coco/tfdata_val"),coco_get_data,90,coco_category_index),
        ("mnistod_train", os.path.join(dataset_root_path,"mnistod/train_tfrecord"),coco_get_data,3,mod_category_index),
        ("mnistod_eval", os.path.join(dataset_root_path,"mnistod/eval_tfrecord"),coco_get_data,3,mod_category_index),
        ("mnistgeood_train", os.path.join(dataset_root_path,"mnistgeood_data/tftrain_big"),coco_get_data,10,modgeo_category_index),
        ("mnistgeood_eval", os.path.join(dataset_root_path,"mnistgeood_data/tftest_big"),coco_get_data,10,modgeo_category_index),
        ("mot_train", os.path.join(dataset_root_path,"MOT/tfdata_mot_train"),mot_get_data,1,default_category_index), #1个类别表示只追踪人
        ("mot_small_train", os.path.join(dataset_root_path,"MOT/tfdata_mot_small_trainv2"),mot_get_data,1,default_category_index), #1个类别表示只追踪人
        ("cell", "/home/wj/ai/mldata2/cell_instance_segmentation/train_tfrecord", coco_get_data, 1, {1:"1"})
    ]
    for x in SPLITS:
        name = x[0]
        args = x[1:]
        DATASETS_REGISTRY.register(name,args)


register_all_pascal_voc()
register_all_coco()
