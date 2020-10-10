#coding=utf-8
from collections import Counter
from collections import Iterable
import wml_utils as wmlu
from iotoolkit.pascal_voc_toolkit import PascalVOCData
import random

def count_file(data):
    img_file, labels = data
    labels_set = set(labels)
    res = {}
    for v in labels_set:
        res[v] = 1
    return res

def count_bboxes(data):
    img_file, labels = data
    labels_set = set(labels)
    return Counter(labels)

def show_dict(values,total_nr):
    print("{")
    for k,v in values.items():
        print(k,":",v,f" {v*100/total_nr:.4f},")
    print("}")

def get_repeat_nr(labels,data_sum_less,target_nr):
    repeat_nr = 0
    for k,v in data_sum_less.items():
        if k in labels:
            nr = target_nr//v
            if nr > repeat_nr:
                repeat_nr = nr
    return repeat_nr

def sample_one_by_key(unused_data,label_to_index,label):
    if label not in label_to_index:
        return None

    if len(label_to_index[label]) == 0:
        label_to_index.pop(label)
        return None

    index = label_to_index[label][0]
    data = unused_data[index]
    for k,v in label_to_index.items():
        if index in v:
            v.remove(index)
    return data

def sample_data(dataset,num_classes,target_nr,count_fn=count_bboxes,extra_file_nr=None,max_repeat_nr=None):
    '''
    num_classes: 类别数量，不包含背景
    '''
    print("Counting data ...")
    data_sum = {}
    datas = []
    for _data in dataset:
        data = _data[0],_data[2]
        count = count_fn(data)
        data_sum = wmlu.add_dict(data_sum,count)
        datas.append(data)

    total_nr = 0
    data_sum_less = {}
    data_sum_greater = {}

    for k,v in data_sum.items():
        total_nr += v
        if v<target_nr:
            data_sum_less[k] = v
        else:
            data_sum_greater[k] = v

    for i in range(1,num_classes+1):
        repeat_nr = get_repeat_nr([i],data_sum_less,target_nr)
        print(f"label {i} repeat {repeat_nr}.")

    print("Count data finish.")
    show_dict(data_sum,total_nr)
    print("Sampling data...")
    sampled_data = []
    unused_data = []
    for i,data in enumerate(datas):
        img_file, labels = data
        repeat_nr = get_repeat_nr(labels,data_sum_less,target_nr)

        if max_repeat_nr is not None:
            repeat_nr = min(repeat_nr,max_repeat_nr)

        if repeat_nr>0:
            sampled_data = sampled_data+[data]*repeat_nr
        else:
            unused_data.append(data)

    tmp_data_sum = {}
    for data in sampled_data:
        count = count_fn(data)
        tmp_data_sum = wmlu.add_dict(tmp_data_sum,count)

    print(f"After repeat")
    total_nr = 0
    for k,v in tmp_data_sum.items():
        total_nr += v

    show_dict(tmp_data_sum,total_nr)


    label_to_index = {}
    for k,v in enumerate(unused_data):
        img_file, labels = v
        labels_set = set(labels)
        for l in labels_set:
            if l in label_to_index:
                label_to_index[l].append(k)
            else:
                label_to_index[l] = [k]

    for k,v in label_to_index.items():
        random.shuffle(v)

    if extra_file_nr is not None:
        total_extra_sample = 0
        for k,v in data_sum_greater.items():
            if k in tmp_data_sum and tmp_data_sum[k]>target_nr:
                count = 0
                for count in range(extra_file_nr):
                    data = sample_one_by_key(unused_data,label_to_index,k)
                    if data is None:
                        print("sample faild.")
                        break
                    count = count_fn(data)
                    tmp_data_sum = wmlu.add_dict(tmp_data_sum,count)
                    sampled_data.append(data)
                    total_extra_sample += 1

        print(f"Total extra sample nr {total_extra_sample}, total sample nr {len(sampled_data)}, After extra sample")
        total_nr = 0
        for k,v in tmp_data_sum.items():
            total_nr += v
    
        show_dict(tmp_data_sum,total_nr)


    while True:
        is_ok = False
        for i in range(1,num_classes+1):
            if i not in tmp_data_sum or tmp_data_sum[i]<target_nr:
                data = sample_one_by_key(unused_data,label_to_index,i)
                if data is None:
                    continue
                is_ok = True
                count = count_fn(data)
                tmp_data_sum = wmlu.add_dict(tmp_data_sum,count)
                sampled_data.append(data)
                #print(f"sample {i}.")
                #show_dict(tmp_data_sum,total_nr)
        if not is_ok:
            break
    print(f"Finish sample, total sample nr {len(sampled_data)}.")
    total_nr = 0
    for k,v in tmp_data_sum.items():
        total_nr += v

    show_dict(tmp_data_sum,total_nr)
    return sampled_data

'''LABELS_NAME = ['BACKGROUND',
    'LSIL',
    'HSIL',
    'TRI',
    'CC',
    'AGC',
    'ACTINO',
    'EC',
    'SCC',
    'CANDIDA',
    'HSV',
]
def labeltext_to_label(name):
    if not isinstance(name,str) and isinstance(name,Iterable):
        return [LABELS_NAME.index(x) for x in name]

    if name not in LABELS_NAME:
        raise ValueError(f"Error show name {name}")
    return LABELS_NAME.index(name)
if __name__ == "__main__":

    data = PascalVOCData(label_text2id=labeltext_to_label)
    data.read_data("/2_data/wj/mldata/cell/stage01_verify_preproc/SLICEID_20200828180746_A_24_1185")
    datas = sample_data(data.get_items(),13,target_nr=300)
    files = []
    for d in datas:
        files.append(wmlu.base_name(d[0]))
    print(len(files))
    wmlu.show_list(files)
    files = list(set(files))
    print("------------------------------------------")
    print(len(files))
    wmlu.show_list(files)'''
