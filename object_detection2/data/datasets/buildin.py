# -*- coding: utf-8 -*-
import os
from .build import DATASETS_REGISTRY
from iotoolkit.pascal_voc_tf_decodev2 import get_data as voc_get_data

# ==== Predefined datasets and splits for COCO ==========
dataset_root_path = "/home/vghost/ai/mldata"

def register_all_pascal_voc(root="datasets"):
    SPLITS = [
        ("voc_2012_train", os.path.join(dataset_root_path,"VOC2012_tfdata"),voc_get_data)
    ]
    for name, dirname,func in SPLITS:
        DATASETS_REGISTRY.register(name,[dirname,func])


register_all_pascal_voc()
