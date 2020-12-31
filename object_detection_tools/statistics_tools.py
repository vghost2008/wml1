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
from iotoolkit.pascal_voc_toolkit import PascalVOCData
from iotoolkit.coco_toolkit import COCOData
from iotoolkit.labelme_toolkit import LabelMeData
import object_detection2.bboxes as odb 
import pandas as pd
import wml_utils as wmlu
from sklearn.cluster import KMeans

'''
ratio: h/w
'''
def statistics_boxes(boxes,nr=100,name=""):
    sizes = [math.sqrt((x[2]-x[0])*(x[3]-x[1])) for x in boxes]
    ratios = [(x[2]-x[0])/(x[3]-x[1]+1e-8) for x in boxes]
    print(f"Min area size {min(sizes)}, max area size {max(sizes)} (pixel).")
    print(f"Min ratio {min(ratios)}, max ratios {max(ratios)}.")
    '''plt.figure(2,figsize=(10,10))
    n, bins, patches = plt.hist(sizes, nr, normed=None, facecolor='blue', alpha=0.5)
    plt.grid(axis='y', alpha=0.75)
    plt.grid(axis='x', alpha=0.75)
    plt.title(name+" area")
    plt.figure(3,figsize=(10,10))
    n, bins, patches = plt.hist(ratios, nr, normed=None, facecolor='red', alpha=0.5)
    plt.grid(axis='y', alpha=0.75)
    plt.grid(axis='x', alpha=0.75)
    plt.title(name+" ratio")
    plt.show()
    print(max(ratios))
    return _statistics_value(sizes,nr),_statistics_value(ratios,nr)'''
    pd_sizes = pd.Series(sizes)
    pd_ratios = pd.Series(ratios)
    plt.figure(0,figsize=(15,10))
    pd_sizes.plot(kind = 'hist', bins = nr, color = 'steelblue', edgecolor = 'black', normed = True, label = "hist")
    pd_sizes.plot(kind = 'kde', color = 'red', label ="kde")
    plt.grid(axis='y', alpha=0.75)
    plt.grid(axis='x', alpha=0.75)
    plt.title(name+" area")

    plt.figure(1,figsize=(15,10))
    pd_ratios.plot(kind = 'hist', bins = nr, color = 'steelblue', edgecolor = 'black', normed = True, label = "hist")
    pd_ratios.plot(kind = 'kde', color = 'red', label ="kde")
    plt.grid(axis='y', alpha=0.75)
    plt.grid(axis='x', alpha=0.75)
    plt.title(name+" ratio")
    plt.show()
    print(max(ratios))
    return _statistics_value(sizes,nr),_statistics_value(ratios,nr)

def statistics_boxes_by_different_area(boxes,nr=100,bin_size=5):
    sizes = [math.sqrt((x[2]-x[0])*(x[3]-x[1])) for x in boxes]
    min_size = min(sizes)
    max_size = max(sizes)
    delta = (max_size-min_size)/bin_size
    l_bboxes = {}
    for i,s in enumerate(sizes):
        index = int((s-min_size)/delta)
        if index in l_bboxes:
            l_bboxes[index].append(boxes[i])
        else:
            l_bboxes[index] = [boxes[i]]

    for k,v in l_bboxes.items():
        print(k,len(v))

    for i in range(bin_size):
        if i not in l_bboxes:
            continue
        l_size = min_size + i* delta
        h_size = l_size + delta
        statistics_boxes(l_bboxes[i],nr,name=f"area_{l_size:.3f}->{h_size:.3f}")

def statistics_boxes_by_different_ratio(boxes,nr=100,bin_size=5):
    ratios = [(x[2]-x[0])/(x[3]-x[1]+1e-8) for x in boxes]
    min_ratio = min(ratios)
    max_ratio = max(ratios)
    delta = (max_ratio-min_ratio)/bin_size
    l_bboxes = {}
    for i, r in enumerate(ratios):
        index = int((r - min_ratio) / delta)
        if index in l_bboxes:
            l_bboxes[index].append(boxes[i])
        else:
            l_bboxes[index] = [boxes[i]]

    for k, v in l_bboxes.items():
        print(k, len(v))
    for i in range(bin_size):
        if i not in l_bboxes:
            continue
        l_ratio= min_ratio+ i * delta
        h_ratio= l_ratio+ delta
        statistics_boxes(l_bboxes[i], nr, name=f"ratio_{l_ratio:.3f}->{h_ratio:.3f}")
    pass

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
        files = wml_utils.recurse_get_filepath_in_dir(dir_path,suffix=".xml")
        print("\ntotal file size {}.".format(len(files)))
        for file in files:
            shape, bboxes, labels_text, difficult, truncated = odu.read_voc_xml(file,aspect_range=aspect_range)
            yield bboxes,labels_text,os.path.basename(file)

    return statistics_boxes_with_datas(get_datas(),label_encoder,labels_to_remove,nr)

def trans_img_short_size_to(img_size,short_size=640):
    img_short_size = min(img_size[0],img_size[1])
    scale = short_size/img_short_size
    return [x*scale for x in img_size]

def trans_img_long_size_to(img_size,long_size=512):
    img_long_size = max(img_size[0],img_size[1])
    scale = long_size/img_long_size
    return [x*scale for x in img_size]

def statistics_boxes_with_datas(datas,label_encoder=default_encode_label,labels_to_remove=None,max_aspect=None,absolute_size=False,
                                trans_img_size=None):
    all_boxes = []
    all_labels = []
    max_examples = 0
    label_file_count={}
    labels_to_file={}
    example_nrs = []
    classeswise_boxes = {}
    total_file_nr = 0

    for data in datas:
        file, img_size,category_ids, labels_text, bboxes, binary_mask, area, is_crowd, _ = data
        total_file_nr += 1
        if bboxes.shape[0]<1:
            print(f"{file} no annotation, skip")
            continue
        if absolute_size:
            if trans_img_size is not None:
                img_size = trans_img_size(img_size)
            bboxes = odb.relative_boxes_to_absolutely_boxes(bboxes,width=img_size[1],height=img_size[0])
        file = os.path.basename(file)
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
        for l,box in zip(labels_text,bboxes):
            if l in classeswise_boxes:
                classeswise_boxes[l].append(box)
            else:
                classeswise_boxes[l] = [box]

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
    labels_counter = list(labels_counter.items())
    labels_counter.sort(key=lambda x:x[1],reverse=True)
    total_nr = 0
    for k,v in labels_counter:
        total_nr += v
    print(f"Total bboxes count {total_nr}")
    print("\n--->BBoxes count:")
    for k,v in labels_counter:
        print("{:>8}:{:<8}, {:>4.2f}%".format(k,v,v*100./total_nr))
    print(f"Total file count {total_file_nr}.")
    print("\n--->File count:")
    label_file_count= list(label_file_count.items())
    label_file_count.sort(key=lambda x:x[1],reverse=True)
    for k,v in label_file_count:
        print("{:>8}:{:<8}, {:>4.2f}".format(k,v,v*100./total_file_nr))
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

    return [all_boxes,classeswise_boxes,labels_to_file]

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

def show_classwise_boxes_statistics(data,nr=20,labels=None):
    for k,v in data.items():
        if labels is not None and not k in labels:
            continue
        statistics_boxes(v,nr=nr,name=k)

def ticks(minv,maxv,order,nr):
    delta = (maxv-minv)/(2.*nr)
    scale = math.pow(10,order)
    n_min = (minv-delta)//scale
    n_max = (maxv+delta)//scale
    minv = n_min*scale
    maxv = n_max*scale
    t_delta = (max(scale,((maxv-minv)/nr))//scale)*scale
    return np.arange(minv,maxv,t_delta).tolist()

def show_anchor_box(img_file,boxes,size=None):
    nr = boxes.shape[0]
    classes = []
    scores = []
    for i in range(nr):
        classes.append(0)
        scores.append(1)
    img = wmli.imread(img_file)
    if size is not None:
        img = wmli.resize_img(img,(size[1],size[0]))

    odv.bboxes_draw_on_img(img, classes, scores, boxes)
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.show()
    return img

def test_dataset():
    data = PascalVOCData(label_text2id=None)
    data.read_data("/home/vghost/ai/mldata2/test_data_0day/test_s")

    return data.get_items()

def pascal_voc_dataset():
    data = PascalVOCData(label_text2id=None)
    #data_path = "/home/wj/ai/mldata3/cell/output/data1"
    #data_path = '/home/wj/ai/mldata3/cell/binary_test_images/pos_cell_for_test5'
    #ata_path = '/home/wj/ai/mldata3/cell/binary_test_images/pos_cell_for_test5/SLICEID_20200903100753_C_15_M1'
    #data_path = '/home/wj/ai/mldata2/cell/annotationed_data/verify_ptp01_1118_p1_proc'
    #data_path = '/home/wj/ai/mldata2/cell/annotationed_data/tmp_test'
    #data_path = '/home/wj/ai/mldata2/cell/annotationed_data/tmp_test'
    #data_path = '/home/wj/ai/mldata3/cell/binary_test_images/pos_cell_for_test20'
    data_path = "/home/wj/ai/mldata3/cell/neg_stage04_preproc/"
    data_path = "/home/wj/ai/mldata2/cell/annotationed_data/tmp_neg_test"
    data_path = "/home/vghost/ai/work/mnistgeood_data_generate/data"
    data_path = "/home/wj/ai/mldata2/cell/annotationed_data/rdatasv5_pos_proc_m"
    data_path = "/0_data/data/user/wbz/OCR/rdatav23/"
    #data_path = '/home/wj/ai/mldata2/cell/annotationed_data/rdatasv4_pos_proc_m'
    #data_path = '/home/vghost/ai/mldata3/cell/pos_stage_01_verify_preprocv3_1'
    #data_path = '/home/wj/ai/mldata3/cell/data2annotation/verify_p04_1105_2'
    #data.read_data("/home/vghost/ai/mldata2/ocrdata/rdatasv20/train")
    #data.read_data("/2_data/wj/mldata/cell/stage01_verify_preproc/")
    #data.read_data('/home/vghost/ai/mldata2/cell/annotationed_data/verify_p03_1020_2_proc')
    #data.read_data('/home/vghost/桌面/verify_p04_1105')
    #data.read_data('/home/vghost/ai/mldata3/cell/output/faild_neg_images_不要标注')
    data.read_data(data_path,silent=True)

    return data.get_items()

def coco2014_dataset():
    data = COCOData()
    data.read_data(wmlu.home_dir("ai/mldata/coco/annotations/instances_train2014.json"), 
                   image_dir=wmlu.home_dir("ai/mldata/coco/train2014"))

    return data.get_items()

def coco2017_dataset():
    data = COCOData()
    data.read_data(wmlu.home_dir("ai/mldata2/coco/annotations/instances_train2017.json"),
                   image_dir=wmlu.home_dir("ai/mldata2/coco/train2017"))

    return data.get_items()

def coco2014_val_dataset():
    data = COCOData()
    data.read_data(wmlu.home_dir("ai/mldata/coco/annotations/instances_val2014.json"),
                   image_dir=wmlu.home_dir("ai/mldata/coco/val2014"))

    return data.get_items()

def labelme_dataset():
    data = LabelMeData(label_text2id=None)
    data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatasv10")
    return data.get_items()

if __name__ == "__main__":
    nr = 100
    def trans_img_long_size(img_size):
        if img_size[0]<img_size[1]:
            k = 512/img_size[1]
        else:
            k = 512 / img_size[0]
        return [k*img_size[0],k*img_size[1]]
    
    def trans_img_short_size(img_size,min_size=640):
        if img_size[0]<img_size[1]:
            k = min_size/img_size[0]
        else:
            k = min_size/ img_size[1]
        return [k*img_size[0],k*img_size[1]]

    statics = statistics_boxes_with_datas(pascal_voc_dataset(),
                                          label_encoder=default_encode_label,
                                          labels_to_remove=None,
                                          max_aspect=None,absolute_size=True,
                                          trans_img_size=None)

    statistics_boxes(statics[0], nr=nr)
    statistics_boxes_by_different_area(statics[0],nr=nr,bin_size=5)
    statistics_boxes_by_different_ratio(statics[0],nr=nr,bin_size=5)
    #show_boxes_statistics(statics)
    show_classwise_boxes_statistics(statics[1],nr=nr)
    '''data = statics[1]
    boxes = []
    for k,v in data.items():
        if len(k)==1:
            boxes.extend(v)
    data['Char'] = boxes
    show_classwise_boxes_statistics(statics[1],nr=nr,labels=["WD0","WD1","WD2","WD3","Char"])
    '''
