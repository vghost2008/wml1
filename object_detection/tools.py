#coding=utf-8
import utils
import os
import utils as odu
import object_detection.npod_toolkit as npod
import wml_utils
import matplotlib.pyplot as plt
import numpy as np
import math

def statistics_boxes(boxes,nr=100):
    sizes = [(x[2]-x[0])*(x[3]-x[1]) for x in boxes]
    ratios = [(x[2]-x[0])/(x[3]-x[1]) for x in boxes]
    print(max(ratios))
    return _statistics_value(sizes,nr),_statistics_value(ratios,nr)

def _statistics_value(values,nr=100):
    value_max = max(values)
    value_min = min(values)
    values_y = [0.]*nr

    for v in values:
        if v<=value_min:
            values_y[0] += 1.0
        elif v>=value_max:
            values_y[nr-1] += 1.0
        else:
            index = int((v-value_min)*(nr-1)/(value_max-value_min))
            values_y[index] += 1.0

    return values_y,value_min,value_max

def default_encode_label(l):
    return l

def statistics_boxes_in_dir(dir_path,label_encoder=default_encode_label,labels_to_remove=None,nr=100,aspect_range=None):
    if not os.path.exists(dir_path):
        print("path {} not exists.".format(dir_path))
    files = wml_utils.recurse_get_filepath_in_dir(dir_path,suffix=".xml")
    all_boxes = []
    all_labels = []
    max_examples = 0
    label_file_count={}
    labels_to_file={}
    for file in files:
        shape, bboxes, labels_text, difficult, truncated = utils.read_voc_xml(file,aspect_range=aspect_range)
        if len(labels_text)==0:
            continue
        aspect = npod.box_aspect(bboxes)
        if np.max(aspect)>6.:
            print("error")
        max_examples = max(len(labels_text),max_examples)
        all_boxes.extend(bboxes)
        all_labels.extend(labels_text)

        tmp_dict = {}
        for l in labels_text:
            tmp_dict[l] = 1
            if labels_to_file.has_key(l):
                labels_to_file[l].append(file)
            else:
                labels_to_file[l] = [file]

        for k in tmp_dict.keys():
            if label_file_count.has_key(k):
                label_file_count[k] += 1
            else:
                label_file_count[k] = 1

    labels_counter = {}
    org_labels_counter = {}
    encoded_labels = []
    for _l in all_labels:
        l = label_encoder(_l)
        encoded_labels.append(l)
        if labels_counter.has_key(l):
            labels_counter[l] = labels_counter[l]+1
        else:
            labels_counter[l] = 1
        if org_labels_counter.has_key(_l):
            org_labels_counter[_l] = org_labels_counter[_l]+1
        else:
            org_labels_counter[_l] = 1
    print("total file size {}.".format(len(files)))
    print("Max element size {}.".format(max_examples))
    print("BBoxes count.")
    for k,v in labels_counter.items():
        print("{}:{}".format(k,v))
    print("File count.")
    for k,v in label_file_count.items():
        print("{}:{}".format(k,v))
    print("org statistics")
    for k,v in org_labels_counter.items():
        print("{}:{}".format(k,v))
    if labels_to_remove is not None:
        all_boxes,encoded_labels = odu.removeLabels(all_boxes,encoded_labels,labels_to_remove)

    res = list(statistics_boxes(all_boxes,nr))
    res.append(labels_to_file)
    return res

def show_boxes_statistics(statics):
    plt.figure(0,figsize=(10,10))
    sizes = statics[0]
    nr = len(sizes[0])
    sizes_x = sizes[1]+np.array(range(nr)).astype(np.float32)*(sizes[2]-sizes[1])/(nr-1)
    sizes_x = sizes_x.tolist()
    plt.title("Size")
    plt.xticks(ticks(sizes[1],sizes[2],-3,20))
    plt.plot(sizes_x,sizes[0])
    plt.figure(1,figsize=(10,10))
    ratios = statics[1]
    nr = len(ratios[0])
    ratios_x = ratios[1]+np.array(range(nr)).astype(np.float32)*(ratios[2]-ratios[1])/(nr-1)
    ratios_x = ratios_x.tolist()
    plt.title("Ratio")
    plt.xticks(ticks(ratios[1],ratios[2],-1,20))
    plt.plot(ratios_x,ratios[0])
    plt.show()

def ticks(minv,maxv,order,nr):
    delta = (maxv-minv)/(2.*nr)
    scale = math.pow(10,order)
    n_min = (minv-delta)//scale
    n_max = (maxv+delta)//scale
    minv = n_min*scale
    maxv = n_max*scale
    t_delta = (max(scale,((maxv-minv)/nr))//scale)*scale
    return np.arange(minv,maxv,t_delta).tolist()


if __name__ == "__main__":
    statics = statistics_boxes_in_dir("/Users/vghost/MachineLearning/mldata/dentalfilm/fullviewod_jpgdatav7",nr=20)
    #statics = statistics_boxes_in_dir("../../../mldata/dentalfilm/diseasedod",nr=10)
    #statics = statistics_boxes_in_dir("../../../mldata/dentalfilm/diseasedod_jpgdatav1/Annotations",nr=10)
    show_boxes_statistics(statics)
