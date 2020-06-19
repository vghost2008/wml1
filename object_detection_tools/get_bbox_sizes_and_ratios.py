#coding=utf-8
from object_detection_tools.statistics_tools import *
from sklearn.cluster import KMeans
import object_detection2.bboxes as odb
import sys
import pickle
import copy

def get_size(bboxes):
    bboxes = np.transpose(bboxes,axes=[1,0])

    w = bboxes[3]-bboxes[1]
    h = bboxes[2]-bboxes[0]
    sizes = np.sqrt(w*h)
    return np.mean(sizes)

def get_ratios(bboxes):
    bboxes = np.transpose(bboxes,axes=[1,0])

    w = bboxes[3]-bboxes[1]
    h = bboxes[2]-bboxes[0]
    ratios = w/(h+1e-8)
    ratios = np.log(ratios)
    r = np.mean(ratios)
    return math.exp(r)

'''
ratio使用w/h与statistics_tools中相反
使用聚类的方法自动确定最佳的anchor box配置
'''
def get_bboxes_sizes_and_ratios_by_kmeans_v1(bboxes,size_nr=3,ratio_nr=3):
    org_bboxes = copy.deepcopy(bboxes)
    bboxes = np.array(bboxes)
    bboxes = np.transpose(bboxes,axes=[1,0])
    cx  = (bboxes[3]+bboxes[1])/2
    cy = (bboxes[2]+bboxes[0])/2
    boxes_center = np.stack([cy,cx,cy,cx],axis=-1)
    new_bboxes = org_bboxes-boxes_center

    w = bboxes[3]-bboxes[1]
    h = bboxes[2]-bboxes[0]
    sizes = np.sqrt(w*h)
    ratios = w/(h+1e-8)
    clf = KMeans(n_clusters=size_nr)
    clf.fit(np.expand_dims(sizes,axis=-1))
    sizes_init = np.squeeze(copy.deepcopy(clf.cluster_centers_),axis=-1)
    clf = KMeans(n_clusters=ratio_nr)
    clf.fit(np.expand_dims(ratios,axis=-1))
    ratios_init = np.squeeze(copy.deepcopy(clf.cluster_centers_),axis=-1)
    
    threshold_for_size = 2.0
    threshold_for_ratio = 0.01
    max_loop = 1000
    step = 0

    while True:
        all_ious = []
        for i in range(size_nr):
            for j in range(ratio_nr):
                size = sizes_init[i]
                ratio = ratios_init[j]
                w = size*math.sqrt(ratio)
                h = size/math.sqrt(ratio)
                box = np.array([[-h/2,-w/2,h/2,w/2]])
                ious = odb.npbboxes_jaccard(bbox_ref=box,bboxes=new_bboxes)
                all_ious.append(ious)

        all_ious = np.stack(all_ious,axis=0)
        index = np.argmax(all_ious,axis=0)
        ratio_index = np.mod(index,ratio_nr)
        size_index = index//ratio_nr
        
        print(f"{step} ------------------------------------")
        new_sizes = copy.deepcopy(sizes_init)
        for i in range(size_nr):
            mask = (size_index==i)
            n_bboxes = new_bboxes[mask]
            n_size = get_size(n_bboxes)
            print(f"Size of {i}: {n_bboxes.shape[0]}/{new_bboxes.shape[0]}({100.0*n_bboxes.shape[0]/new_bboxes.shape[0]:.2f}%), old_size={sizes_init[i]}, new_size={n_size}")
            new_sizes[i] = n_size

        new_ratios = copy.deepcopy(ratios_init)
        for i in range(ratio_nr):
            mask = (ratio_index==i)
            n_bboxes = new_bboxes[mask]
            n_ratio = get_ratios(n_bboxes)
            print(f"Ratio of {i}: {n_bboxes.shape[0]}/{new_bboxes.shape[0]}({100.0*n_bboxes.shape[0]/new_bboxes.shape[0]:.2f}%), old_ratio={ratios_init[i]}, new_size={n_ratio}")
            new_ratios[i] = n_ratio
        
        step += 1
        if (np.any(np.fabs(new_sizes-sizes_init)>threshold_for_size) or \
            np.any(np.fabs(new_ratios-ratios_init)>threshold_for_ratio)) and \
            step<max_loop:
            sizes_init = copy.deepcopy(new_sizes)
            ratios_init = copy.deepcopy(new_ratios)
            continue
        break
    vis_data = []
    new_ratios = new_ratios.tolist()
    new_ratios.sort()
    for i in range(size_nr):
        vis_data.append((new_sizes[i],new_ratios))
    vis_data.sort(key=lambda x:x[0])
    str0,str1 = get_formated_string(vis_data)
    print(str0)
    print(str1)
    
    sizes,ratios = zip(*vis_data)
    return sizes,ratios


'''
ratio使用w/h与statistics_tools中相反
使用聚类的方法自动确定最佳的anchor box配置
与v1的差别为不同的size使用不同的ratio
'''
def get_bboxes_sizes_and_ratios_by_kmeans_v2(bboxes, size_nr=3, ratio_nr=3,group_size=None):
    org_bboxes = copy.deepcopy(bboxes)
    bboxes = np.array(bboxes)
    bboxes = np.transpose(bboxes, axes=[1, 0])
    cx = (bboxes[3] + bboxes[1]) / 2
    cy = (bboxes[2] + bboxes[0]) / 2
    boxes_center = np.stack([cy, cx, cy, cx], axis=-1)
    new_bboxes = org_bboxes - boxes_center

    w = bboxes[3] - bboxes[1]
    h = bboxes[2] - bboxes[0]
    sizes = np.sqrt(w * h)
    #sizes = np.array([0.9,1.0,1.1,1.9,2.0,2.1,2.9,3.0,3.1,3.9,4.0,4.1])
    ratios = w / (h + 1e-8)
    clf = KMeans(n_clusters=size_nr)
    clf.fit(np.expand_dims(sizes, axis=-1))
    sizes_init = np.squeeze(copy.deepcopy(clf.cluster_centers_), axis=-1)
    all_dis = []
    for i in range(size_nr):
        dis = np.fabs(sizes-sizes_init[i])
        all_dis.append(dis)
    all_dis = np.stack(all_dis,axis=0)
    size_index = np.argmin(all_dis,axis=0)
    ratios_init = []
    for i in range(size_nr):
        mask = (size_index==i)
        clf = KMeans(n_clusters=ratio_nr)
        clf.fit(np.expand_dims(ratios[mask], axis=-1))
        ratio_init = np.squeeze(copy.deepcopy(clf.cluster_centers_), axis=-1)
        ratios_init.append(ratio_init)

    ratios_init = np.stack(ratios_init,axis=0)

    threshold_for_size = 2.0
    threshold_for_ratio = 0.01
    max_loop = 500
    step = 0

    while True:
        all_ious = []
        for i in range(size_nr):
            for j in range(ratio_nr):
                size = sizes_init[i]
                ratio = ratios_init[i][j]
                w = size * math.sqrt(ratio)
                h = size / math.sqrt(ratio)
                box = np.array([[-h / 2, -w / 2, h / 2, w / 2]])
                ious = odb.npbboxes_jaccard(bbox_ref=box, bboxes=new_bboxes)
                all_ious.append(ious)

        all_ious = np.stack(all_ious, axis=0)
        index = np.argmax(all_ious, axis=0)
        ratio_index = np.mod(index, ratio_nr)
        size_index = index // ratio_nr

        print(f"{step} ------------------------------------")
        new_sizes = copy.deepcopy(sizes_init)
        new_ratios = copy.deepcopy(ratios_init)
        for i in range(size_nr):
            mask = (size_index == i)
            n_bboxes = new_bboxes[mask]
            n_size = get_size(n_bboxes)
            print(
                f"\nSize of {i}: {n_bboxes.shape[0]}/{new_bboxes.shape[0]}({100.0 * n_bboxes.shape[0] / new_bboxes.shape[0]:.2f}%), old_size={sizes_init[i]}, new_size={n_size}")
            new_sizes[i] = n_size
            for j in range(ratio_nr):
                r_mask = np.logical_and((ratio_index == j),mask)
                n_bboxes = new_bboxes[r_mask]
                n_ratio = get_ratios(n_bboxes)
                print(
                    f"Ratio of {j}: {n_bboxes.shape[0]}/{new_bboxes.shape[0]}({100.0 * n_bboxes.shape[0] / new_bboxes.shape[0]:.2f}%), old_ratio={ratios_init[i][j]}, new_ratios={n_ratio}")
                new_ratios[i][j] = n_ratio

        step += 1
        if (np.any(np.fabs(new_sizes - sizes_init) > threshold_for_size) or \
            np.any(np.fabs(new_ratios - ratios_init) > threshold_for_ratio)) and \
                step < max_loop:
            sizes_init = copy.deepcopy(new_sizes)
            ratios_init = copy.deepcopy(new_ratios)
            continue
        break
    vis_data = []
    for i in range(size_nr):
        d = new_ratios[i].tolist()
        d.sort()
        vis_data.append((new_sizes[i],d))
    vis_data.sort(key=lambda x:x[0])
    str0,str1 = get_formated_string(vis_data,group_size)
    print(str0)
    print(str1)
    sizes,ratios = zip(*vis_data)
    return sizes,ratios

'''
ratio使用w/h与statistics_tools中相反
使用聚类的方法自动确定最佳的anchor box配置
与v2的差别为增加了一组宽高颠倒的boxes
'''
def get_bboxes_sizes_and_ratios_by_kmeans_v3(bboxes, size_nr=3, ratio_nr=3,group_size=None):
    bboxes = np.array(bboxes)
    ymin = bboxes[:,0]
    xmin = bboxes[:,1]
    ymax = bboxes[:,2]
    xmax = bboxes[:,3]
    new_ymin = np.concatenate([ymin,xmin],axis=0)
    new_xmin = np.concatenate([xmin,ymin],axis=0)
    new_ymax = np.concatenate([ymax,xmax],axis=0)
    new_xmax = np.concatenate([xmax,ymax],axis=0)
    new_bboxes = np.stack([new_ymin,new_xmin,new_ymax,new_xmax],axis=-1)
    return get_bboxes_sizes_and_ratios_by_kmeans_v2(new_bboxes,size_nr=size_nr,ratio_nr=ratio_nr,group_size=group_size)

def list_to_str(data,add_brackets=False):
    res = "["
    for d in data:
        if add_brackets:
            res+=f"[{d:.2f}],"
        else:
            res+=f"{d:.2f},"
    res = res[:-1]
    res += "]"
    return res
    
def list2d_to_str(data):
    str1 = "["
    for d in data:
        str1 += list_to_str(d) + ","
    str1 = str1[:-1]
    str1 += "]"
    return str1

def list3d_to_str(data):
    str1 = "["
    for d in data:
        str1 += list2d_to_str(d) + ","
    str1 = str1[:-1]
    str1 += "]"
    return str1

def get_formated_string(vis_data,group_size=None):
    sizes,ratios = zip(*vis_data)
    if group_size is None:
        str0 = list_to_str(sizes,add_brackets=True)
        str1 = list2d_to_str(ratios)
    else:
        sizes = wmlu.list_to_2dlist(sizes,group_size)
        ratios = wmlu.list_to_2dlist(ratios,group_size)
        str0 = list2d_to_str(sizes)
        str1 = list3d_to_str(ratios)

    return str0,str1

if len(sys.argv)>=2:
    data_path = sys.argv[-1]
else:
    data_path = wmlu.home_dir("ai/mldata2/0day/bbox1.dat")
    data_path = wmlu.home_dir("ai/mldata2/0day/coco_statics.dat")
    #data_path = wmlu.home_dir("ai/mldata2/0day/ocr_statics.dat")
    # data_path = wmlu.home_dir("ai/mldata2/0day/qc_statics.dat")

print(f"Data path {data_path}")

if __name__ == "__main__":
    with open(data_path,"rb") as file:
        statics = pickle.load(file)
    nr = 100
    boxes = statics[0]
    new_sizes,new_ratios = get_bboxes_sizes_and_ratios_by_kmeans_v2(bboxes=boxes,size_nr=10,ratio_nr=3,group_size=2)
