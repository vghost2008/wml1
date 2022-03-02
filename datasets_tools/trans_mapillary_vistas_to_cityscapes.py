import sys

from iotoolkit.mapillary_vistas_toolkit import *
from multiprocess import Pool
import img_utils as wmli
import object_detection_tools.visualization as odv
import matplotlib.pyplot as plt
import numpy as np
import object_detection2.mask as odm
import wml_utils as wmlu
import copy
import json
import cv2

lid = 0

name_to_id_dict = {
"construction--flat--bike-lane":0,
"construction--flat--driveway":0,
"construction--flat--road":0,
"construction--flat--road-shoulder":0,
"construction--flat--rail-track":0,
"construction--flat--sidewalk":1,
"object--street-light":2,
"construction--structure--bridge":3,
"construction--structure--building":4,
"human--":5,
"object--support--pole":6,
"marking--continuous--dashed":7,
"marking--continuous--solid":8,
"marking--discrete--crosswalk-zebra":9,
"nature--sand":10,
"nature--sky":11,
"nature--snow":12,
"nature--terrain":13,
"nature--vegetation":14,
"nature--water":15,
"object--vehicle--bicycle":16,
"object--vehicle--boat":17,
"object--vehicle--bus":18,
"object--vehicle--car":19,
"object--vehicle--vehicle-group":19,
"object--vehicle--caravan":20,
"object--vehicle--motorcycle":21,
"object--vehicle--on-rails":22,
"object--vehicle--truck":23,
"construction--flat--pedestrian-area":24,
"construction--structure--tunnel":25,
"void--ground":26,
"nature--":255,
"construction--":255,
"object--bench":255,
"void--":255,
}

def update_name_to_id(dict_data,dir):
    names = []
    with open(os.path.join(dir,"config_v2.0.json")) as fp:
        data = json.load(fp)
        data = data['labels']
        for x in data:
            names.append(x['name'])
    new_dict_data = {}
    for k,v in dict_data.items():
        if k.endswith("--"):
            for name in names:
                if name.startswith(k) and name not in dict_data and name not in new_dict_data:
                    new_dict_data[name] = v
        else:
            new_dict_data[k] = v
    return new_dict_data

def trans_data(data_dir,save_dir,beg,end):
    global name_to_id_dict
    wmlu.show_dict(name_to_id_dict)
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)

    def name_to_id(x):
        return name_to_id_dict[x]

    ignored_labels = ["construction--barrier--ambiguous","construction--barrier--separator"]
    data = MapillaryVistasData(label_text2id=name_to_id, shuffle=False,
                               ignored_labels=ignored_labels,
                               label_map=None,
                               sub_dir_name="validation",
                               #sub_dir_name="training",
                               allowed_labels_fn=list(name_to_id_dict.keys()))
    data.read_data(data_dir)

    def filter(full_path,_):
        base_name = wmlu.base_name(full_path)+".png"
        save_path = os.path.join(save_dir,base_name)
        if os.path.exists(save_path):
            print(f"File {save_path} exists.")
            return False
        print(f"File {save_path} not exists.")
        return True

    for i,x in enumerate(data.get_items(beg,end,filter=filter)):
        full_path, img_info, category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x

        if len(category_ids) == 0:
            print(f"Skip {full_path}")
            continue

        new_mask = odm.dense_mask_to_sparse_mask(binary_mask,category_ids,default_label=255)
        base_name = wmlu.base_name(full_path)+".png"
        save_path = os.path.join(save_dir,base_name)
        new_mask = new_mask.astype(np.uint8)
        if os.path.exists(save_path):
            print(f"WARNING: File {save_path} exists.")
        wmli.imwrite_mask(save_path,new_mask)
        sys.stdout.write(f"\r{i}")


if __name__ == "__main__":
    data_dir ="/home/wj/ai/mldata/mapillary_vistas/"
    save_dir = os.path.join(data_dir,'boe_labels_validation')
    name_to_id_dict = update_name_to_id(name_to_id_dict,data_dir)
    idxs = list(range(0,18049,50))
    r_idxs = []
    for i in range(len(idxs)-1):
        r_idxs.append([idxs[i],idxs[i+1]])
    wmlu.show_list(r_idxs)
    pool = Pool(10)
    def fun(d):
        trans_data(data_dir,save_dir,d[0],d[1])
    res = list(pool.map(fun,r_idxs))
    pool.close()
    pool.join()
    print(res)
    #list(map(fun,r_idxs))

