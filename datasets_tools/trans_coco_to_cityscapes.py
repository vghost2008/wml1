import sys

from iotoolkit.coco_toolkit import *
import shutil
import img_utils as wmli
import object_detection_tools.visualization as odv
import matplotlib.pyplot as plt
import numpy as np
import object_detection2.mask as odm
import wml_utils as wmlu
import copy
import json
import cv2

#resize_size = (2560,1440)
resize_size = None

def trans_data(annotations_file,image_dir,save_dir,copy_img=True):
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)

    def trans_label(x):
        trans_dict = {1:0,32:1}
        if x in trans_dict:
            return trans_dict[x]
        return None

    data = COCOData(trans_label)
    data.read_data(annotations_file,image_dir)
    for i,x in enumerate(data.get_items()):
        full_path, img_info, category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x

        if len(category_ids) == 0:
            print(f"Skip {full_path}")
            continue

        new_mask = odm.dense_mask_to_sparse_maskv2(binary_mask,category_ids,labels_order=[0,1],default_label=255)
        #r_base_name = f"IMG_{i+1:05d}"
        r_base_name = wmlu.base_name(full_path)
        base_name = r_base_name+".png"
        save_path = os.path.join(save_dir,base_name)
        if resize_size is not None:
            new_mask = wmli.resize_img(new_mask,resize_size,keep_aspect_ratio=True,interpolation=cv2.INTER_NEAREST)
            img  = wmli.imread(full_path)
            img = wmli.resize_img(img,resize_size,keep_aspect_ratio=True)
            img_save_path = os.path.join(save_dir,r_base_name+".jpg")
            wmli.imwrite(img_save_path,img)
        else:
            img_save_path = os.path.join(save_dir,r_base_name+".jpg")
            shutil.copy(full_path,img_save_path)

        new_mask = new_mask.astype(np.uint8)
        if os.path.exists(save_path):
            print(f"WARNING: File {save_path} exists.")
        wmli.imwrite_mask(save_path,new_mask)
        sys.stdout.write(f"\r{i}")

if __name__ == "__main__":
    annotations_file = '/home/wj/ai/mldata/coco/annotations/instances_train2017.json'
    image_dir = '/home/wj/ai/mldata/coco/train2017'
    save_dir = os.path.join("/home/wj/ai/mldata1/safety_belt/",'boe_labels_train')
    trans_data(annotations_file,image_dir,save_dir)

    annotations_file = '/home/wj/ai/mldata/coco/annotations/instances_val2017.json'
    image_dir = '/home/wj/ai/mldata/coco/val2017'
    save_dir = os.path.join("/home/wj/ai/mldata1/safety_belt/",'boe_labels_val')
    trans_data(annotations_file,image_dir,save_dir)
