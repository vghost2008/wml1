# -*- coding: utf-8 -*-
import os
from .build import DATASETS_REGISTRY
from iotoolkit.pascal_voc_tf_decodev2 import get_data as voc_get_data
from iotoolkit.coco_tf_decodev2 import get_data as coco_get_data

# ==== Predefined datasets and splits for COCO ==========
dataset_root_path = "/home/vghost/ai/mldata"

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
        ("coco_2017_train", os.path.join(dataset_root_path,"coco_tfdata"),coco_get_data,90)
    ]
    for x in SPLITS:
        name = x[0]
        args = x[1:]
        DATASETS_REGISTRY.register(name,args)


register_all_pascal_voc()
register_all_coco()
