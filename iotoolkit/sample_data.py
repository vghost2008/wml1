#coding=utf-8
from collections import Counter
from collections import Iterable
import wml_utils as wmlu
from iotoolkit.pascal_voc_toolkit import PascalVOCData
import random
def name_dict_nr2id_dict_nr(data,name_to_id,scale=1):
    res = {}
    for k,v in data.items():
        id = name_to_id(k)
        res[id] = int(v*scale)
    return res

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
    if isinstance(target_nr,(int,float)):
        for k,v in data_sum_less.items():
            if k in labels:
                nr = target_nr//v
                if nr > repeat_nr:
                    repeat_nr = nr
        return repeat_nr
    else:
        for k,v in data_sum_less.items():
            if k in labels:
                nr = target_nr[k]//v
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

def sample_data(dataset,num_classes,target_nr_or_dict_nr,count_fn=count_bboxes,extra_file_nr=None,max_repeat_nr=None):
    '''
    target_nr_or_dict_nr:int/dict key=calsses_id, value=nr e.g.{1:10,2:100,...}
    num_classes: 类别数量，不包含背景
    extra_file_nr: 不需要过采样数据的基本采样次数，None表示不做特别处理
    max_repeat_nr: 重复次数, 0表示不重复
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

    def target_nr(k):
        if isinstance(target_nr_or_dict_nr,(int,float)):
            return target_nr_or_dict_nr
        elif k not in target_nr_or_dict_nr:
            return 0
        else:
            return target_nr_or_dict_nr[k]

    for k,v in data_sum.items():
        total_nr += v
        if v<target_nr(k):
            data_sum_less[k] = v
        else:
            data_sum_greater[k] = v

    for i in range(1,num_classes+1):
        repeat_nr = get_repeat_nr([i],data_sum_less,target_nr(i))
        print(f"label {i} expected to repeat {repeat_nr} times.")

    print("Count data finish.")
    show_dict(data_sum,total_nr)
    print("Sampling data...")
    sampled_data = []
    unused_data = []
    #对数据过采样
    if max_repeat_nr is not None and max_repeat_nr>0:
        for i,data in enumerate(datas):
            img_file, labels = data
            repeat_nr = get_repeat_nr(labels,data_sum_less,target_nr_or_dict_nr)
    
            if max_repeat_nr is not None:
                repeat_nr = min(repeat_nr,max_repeat_nr)
    
            if repeat_nr>0:
                sampled_data = sampled_data+[data]*repeat_nr
            else:
                unused_data.append(data)
    else:
        unused_data = datas

    tmp_data_sum = {}
    for data in sampled_data:
        count = count_fn(data)
        tmp_data_sum = wmlu.add_dict(tmp_data_sum,count)

    print(f"Over sampled data info.")
    #统计并显示过采样数据的相关信息
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

    if extra_file_nr is not None and extra_file_nr>0:
        total_extra_sample = 0
        for k,v in data_sum_greater.items():
            '''
            仅对过采样时意外采到，后继不会再采样的数据进行额外采样
            '''
            if k in tmp_data_sum and tmp_data_sum[k]>target_nr(k):
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
            if i not in tmp_data_sum or tmp_data_sum[i]<target_nr(i):
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
