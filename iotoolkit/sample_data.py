#coding=utf-8
from collections import Counter
from collections import Iterable
import wml_utils as wmlu
from iotoolkit.pascal_voc_toolkit import PascalVOCData
import random
import copy

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

def sort_by_labels_nr(indexs:list,datas:list):
    if len(indexs)<=1:
        return indexs
    new_indexs = [(x,len(datas[x][1])) for x in indexs]
    new_indexs.sort(key=lambda x:x[1],reverse=True)
    return [x[0] for x in new_indexs]

def sort_partial_by_labels_nr(indexs:list,datas:list):
    if len(indexs)<=2:
        return indexs
    nr = len(indexs)
    h_nr = nr//2
    random.shuffle(indexs)
    indexs0 = indexs[:h_nr]
    indexs1 = indexs[h_nr:]
    new_indexs = [(x,len(datas[x][1])) for x in indexs1]
    new_indexs.sort(key=lambda x:x[1],reverse=True)
    indexs1 = [x[0] for x in new_indexs]
    res = []
    for i in range(h_nr):
        res.append(indexs0[i])
        res.append(indexs1[i])
    if len(indexs1)>h_nr:
        res.append(indexs1[-1])
    
    return res

'''
用于限制在每一个tmap中采样的数量
'''
class LimitPatchsNrInSlide(object):
    def __init__(self,get_file_id_fn,max_nr=10):
        self.get_file_id_fn = get_file_id_fn
        self.max_nr = max_nr

    def __call__(self,data,sampled_data,unused_data,label_to_index):
        data_nr = 0
        cur_id = self.get_file_id_fn(data[0])

        for path,labels in sampled_data:
            id = self.get_file_id_fn(path)
            if id == cur_id:
                data_nr += 1

        if data_nr>=self.max_nr:
            for k, v in label_to_index.items():
                for index in copy.deepcopy(v):
                    tmp_path,_ = unused_data[index]
                    id = self.get_file_id_fn(tmp_path)
                    if id == cur_id:
                        v.remove(index)

def sample_data(dataset,num_classes,target_nr_or_dict_nr,count_fn=count_bboxes,extra_file_nr=None,max_repeat_nr=None,
                sort_data_fn=None,
                call_back_after_sampled=None):
    '''
    target_nr_or_dict_nr:int/dict key=calsses_id, value=nr e.g.{1:10,2:100,...}
    num_classes: 类别数量，不包含背景
    extra_file_nr: 不需要过采样数据的基本采样次数，None表示不做特别处理
    max_repeat_nr: 重复次数, 0表示不重复
    call_back_after_sampled(sampled_data:list((file_path,labels)),unused_data:list((file_path,labels)),indexs:dict(key=label,value=indexs))
    '''
    print("Counting data ...")
    data_sum = {}
    datas = []
    for _data in dataset:
        data = _data[0],_data[2]  #0为文件名，2为labels
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

    label_to_index = {} #key为目标标签，v为相应数据在unused_data中的索引
    for k,v in enumerate(unused_data):
        img_file, labels = v
        labels_set = set(labels)
        for l in labels_set:
            if l in label_to_index:
                label_to_index[l].append(k)
            else:
                label_to_index[l] = [k]

    if sort_data_fn is None:
        for k,v in label_to_index.items():
            random.shuffle(v)
    else:
        tmp_label_to_index = copy.deepcopy(label_to_index)
        label_to_index = {}
        for k,v in tmp_label_to_index.items():
            label_to_index[k] = sort_data_fn(v,unused_data)

    if extra_file_nr is not None and extra_file_nr>0:
        total_extra_sample = 0
        for k,v in data_sum_greater.items():
            '''
            仅对过采样时意外采到，后继不会再采样的数据进行额外采样
            '''
            if k in tmp_data_sum and tmp_data_sum[k]>target_nr(k):
                for count in range(extra_file_nr):
                    data = sample_one_by_key(unused_data,label_to_index,k)
                    if data is None:
                        print("sample faild.")
                        break
                    count = count_fn(data)
                    tmp_data_sum = wmlu.add_dict(tmp_data_sum,count)
                    sampled_data.append(data)
                    if call_back_after_sampled is not None:
                        call_back_after_sampled(data,sampled_data,unused_data,label_to_index)
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
                if call_back_after_sampled is not None:
                    call_back_after_sampled(data, sampled_data, unused_data, label_to_index)
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
