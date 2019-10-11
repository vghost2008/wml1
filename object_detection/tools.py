#coding=utf-8
import os
import object_detection.utils as odu
import object_detection.npod_toolkit as npod
import wml_utils
import matplotlib.pyplot as plt
import numpy as np
import math
import object_detection.visualization as odv
import img_utils as wmli

def statistics_boxes(boxes,nr=100):
    sizes = [(x[2]-x[0])*(x[3]-x[1]) for x in boxes]
    ratios = [(x[2]-x[0])/(x[3]-x[1]) for x in boxes]
    plt.figure(0,figsize=(10,10))
    n, bins, patches = plt.hist(sizes, nr*10, normed=None, facecolor='blue', alpha=0.5)
    plt.title("Size")
    plt.figure(1,figsize=(10,10))
    n, bins, patches = plt.hist(ratios, nr*10, normed=None, facecolor='red', alpha=0.5)
    plt.title("Ratio")
    plt.show()
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
    def get_datas():
        if not os.path.exists(dir_path):
            print("path {} not exists.".format(dir_path))
        files = wml_utils.recurse_get_filepath_in_dir(dir_path,suffix=".xml")[:2000]
        print("\ntotal file size {}.".format(len(files)))
        for file in files:
            shape, bboxes, labels_text, difficult, truncated = odu.read_voc_xml(file,aspect_range=aspect_range)
            yield bboxes,labels_text,os.path.basename(file)

    return statistics_boxes_with_datas(get_datas(),label_encoder,labels_to_remove,nr)

def statistics_boxes_with_datas(datas,label_encoder=default_encode_label,labels_to_remove=None,nr=100,max_aspect=None):
    all_boxes = []
    all_labels = []
    max_examples = 0
    label_file_count={}
    labels_to_file={}
    example_nrs = []
    for data in datas:
        bboxes,labels_text,file = data
        if len(labels_text)==0:
            continue
        aspect = npod.box_aspect(bboxes)
        if max_aspect is not None and np.max(aspect)>max_aspect:
            print(f"asepct is too large, expect max aspect is {max_aspect}, actual get {np.max(aspect)}")
        e_nr = len(labels_text)
        example_nrs.append(e_nr)
        max_examples = max(e_nr,max_examples)
        all_boxes.extend(bboxes)
        all_labels.extend(labels_text)

        tmp_dict = {}
        for l in labels_text:
            tmp_dict[l] = 1
            if l in labels_to_file:
                labels_to_file[l].append(file)
            else:
                labels_to_file[l] = [file]

        for k in tmp_dict.keys():
            if k in label_file_count:
                label_file_count[k] += 1
            else:
                label_file_count[k] = 1

    labels_counter = {}
    org_labels_counter = {}
    encoded_labels = []
    for _l in all_labels:
        l = label_encoder(_l)
        encoded_labels.append(l)
        if l in labels_counter:
            labels_counter[l] = labels_counter[l]+1
        else:
            labels_counter[l] = 1
        if _l in org_labels_counter:
            org_labels_counter[_l] = org_labels_counter[_l]+1
        else:
            org_labels_counter[_l] = 1
    example_nrs = np.array(example_nrs)
    print(f"Max element size {np.max(example_nrs)}, element min {np.min(example_nrs)}, element mean {np.mean(example_nrs)}, element var {np.var(example_nrs)}.")
    print("\n--->BBoxes count:")
    labels_counter = list(labels_counter.items())
    labels_counter.sort(key=lambda x:x[1],reverse=True)
    total_nr = 0
    for k,v in labels_counter:
        total_nr += v
    for k,v in labels_counter:
        print("{:>8}:{:<8}, {:>4.2f}%".format(k,v,v*100./total_nr))
    print("\n--->File count:")
    label_file_count= list(label_file_count.items())
    label_file_count.sort(key=lambda x:x[1],reverse=True)
    for k,v in label_file_count:
        print("{:>8}:{:<8}".format(k,v))
    print("\n--->org statistics:")
    org_labels_counter= list(org_labels_counter.items())
    org_labels_counter.sort(key=lambda x:x[1],reverse=True)
    total_nr = 0
    for k,v in org_labels_counter:
        total_nr += v
    for k,v in org_labels_counter:
        print(f"{k:>8}:{v:<8}, {v*100./total_nr:>4.2f}%")
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

def show_anchor_box(img_file,boxes):
    nr = boxes.shape[0]
    classes = []
    scores = []
    for i in range(nr):
        classes.append(0)
        scores.append(1)
    img = wmli.imread(img_file)
    odv.plt_bboxes(img,classes,scores,boxes,show_text=False)

if __name__ == "__main__":
    #statics = statistics_boxes_in_dir("/home/vghost/ai/mldata/udacity/voc/VOC2012",nr=20)
    #statics = statistics_boxes_in_dir("/home/vghost/ai/mldata/ocrdatav1/rdatasvx2/train",nr=20)
    statics = statistics_boxes_in_dir("/home/vghost/ai/mldata/ocrdatav1/rdatavx3",nr=20)
    #statics = statistics_boxes_in_dir("../../../mldata/dentalfilm/diseasedod",nr=10)
    #statics = statistics_boxes_in_dir("../../../mldata/dentalfilm/diseasedod_jpgdatav1/Annotations",nr=10)
    show_boxes_statistics(statics)
