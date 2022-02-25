import sys
import os
import wml_utils as wmlu
import cv2
import numpy as np
from iotoolkit.pascal_voc_toolkit import write_voc_xml

default_classes_names = {}
for i in range(1000):
    default_classes_names[i] = str(i)

def read_yolotxt(txt_path,img_suffix="jpg"):
    labels = []
    bboxes = []
    with open(txt_path,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(" ")
            labels.append(int(line[0]))
            cx = float(line[1])
            cy = float(line[2])
            w = float(line[3])
            h = float(line[4])
            xmin = cx-w/2
            ymin = cy-h/2
            xmax = cx+w/2
            ymax = cy+h/2
            bboxes.append([ymin,xmin,ymax,xmax])
    return np.array(labels),np.array(bboxes)


def trans_yolotxt(txt_path,classes_names,img_suffix="jpg"):
    labels,bboxes = read_yolotxt(txt_path)
    img_path = wmlu.change_suffix(txt_path,img_suffix)
    xml_path = wmlu.change_suffix(txt_path,"xml")
    _bboxes = []
    _labels = []
    for i,x in enumerate(labels):
        if x not in classes_names:
            continue
        _labels.append(classes_names[x])
        _bboxes.append(bboxes[i])
    if len(_labels) == 0:
        print(f"{txt_path} is empty.")
        return
    write_voc_xml(xml_path,img_path,None,_bboxes,_labels)

def trans_dirs(dir_path,classes_names):
    txt_paths = wmlu.recurse_get_filepath_in_dir(dir_path,suffix=".txt")
    for txt_path in txt_paths:
        trans_yolotxt(txt_path,classes_names)

if __name__ == "__main__":
    #classes_names = ["car", "truck", "tank_truck", "bus", "van", "dangerous_sign"]
    classes_names = {1:'car',2:'bus',3:'truck'}
    trans_dirs("/mnt/data1/wj/ai/mldata/boedcvehicle/train_data/txj_vd_bus_truck",classes_names)

