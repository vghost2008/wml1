import wml_utils as wmlu
import os
import json
import numpy as np
import cv2 as cv
import object_detection.bboxes as odb
import object_detection.visualization as odv
import copy
import img_utils as wmli
import random
import matplotlib.pyplot as plt

def get_files(data_dir, img_suffix="jpg"):
    files = wmlu.recurse_get_filepath_in_dir(data_dir, suffix=".json")
    res = []
    for file in files:
        img_file = wmlu.change_suffix(file, img_suffix)
        if os.path.exists(img_file):
            res.append((img_file, file))

    return res

def read_labelme_data(file_path,label_text_to_id=lambda x:int(x)):
    annotations_list = []
    image = {}
    with open(file_path,"r",encoding="gb18030") as f:
        data_str = f.read()
        json_data = json.loads(data_str)
        img_width = json_data["imageWidth"]
        img_height = json_data["imageHeight"]
        image["height"] = img_height
        image["width"] = img_width
        image["file_name"] = wmlu.base_name(file_path)
        for shape in json_data["shapes"]:
            mask = np.zeros(shape=[img_height,img_width],dtype=np.uint8)
            all_points = np.array([shape["points"]])
            points = np.transpose(all_points[0])
            x,y = np.vsplit(points,2)
            xmin = np.min(x)
            xmax = np.max(x)
            ymin = np.min(y)
            ymax = np.max(y)
            segmentation = cv.drawContours(mask,all_points,-1,color=(1),thickness=cv.FILLED)
            label = label_text_to_id(shape["label"])
            annotations_list.append({"bbox":(xmin,ymin,xmax-xmin+1,ymax-ymin+1),"segmentation":segmentation,"category_id":label})
    return image,annotations_list

def save_labelme_data(file_path,image_path,image,annotations_list,label_to_text=lambda x:str(x)):
    data={}
    shapes = []
    data["version"] = "3.10.1"
    data["flags"] = {}
    for ann in annotations_list:
        shape = {}
        shape["label"] = label_to_text(ann["category_id"])
        #shape["line_color"]=None
        #shape["fill_color"]=None
        shape["shape_type"]="polygon"
        contours, hierarchy = cv.findContours(ann["segmentation"], cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for cont in contours:
            points = cont
            if len(cont.shape)==3 and cont.shape[1]==1:
                points = np.squeeze(points,axis=1)
            points = points.tolist()
            shape["points"] = points
            shapes.append(shape)
    data["shapes"] = shapes
    data["imagePath"] = os.path.basename(image_path)
    data["imageWidth"] = image["width"]
    data["imageHeight"] = image["height"]
    with open(file_path,"w") as f:
        json.dump(data,f)

def get_labels_and_bboxes(image,annotations_list):
    labels = []
    bboxes = []
    width = image["width"]
    height = image["height"]
    for ann in annotations_list:
        t_box = ann["bbox"]
        xmin = t_box[0]/width
        ymin = t_box[1]/height
        xmax = xmin+t_box[2]/width
        ymax = ymin+t_box[3]/width
        bboxes.append([ymin,xmin,ymax,xmax])
        labels.append(ann["category_id"])
    return np.array(labels),np.array(bboxes)

'''
output:
[num_classes,h,w] or [num_classes-1,h,w], value is 0 or 1
'''
def get_masks(image,annotations_list,num_classes,no_background=True):
    width = image["width"]
    height = image["height"]
    if no_background:
        get_label = lambda x:max(0,x-1)
        res = np.zeros([num_classes-1,height,width],dtype=np.int32)
    else:
        get_label = lambda x:x
        res = np.zeros([num_classes,height,width],dtype=np.int32)

    for ann in annotations_list:
        mask = ann["segmentation"]
        label = get_label(ann["category_id"])
        res[label:label+1,:,:] = res[label:label+1,:,:]|np.expand_dims(mask,axis=0)

    return res

def get_image_size(image):
    width = image["width"]
    height = image["height"]
    return (height,width)

def save_labelme_datav1(file_path,image_path,image,image_data,annotations_list,label_to_text=lambda x:str(x)):
    wmli.imsave(image_path,image_data)
    save_labelme_data(file_path,image_path,image,annotations_list,label_to_text)

def get_expand_bboxes_in_annotations(annotations,scale=2):
   bboxes = [ann["bbox"] for ann in annotations]
   bboxes = odb.expand_bbox(bboxes,scale)
   return bboxes

def get_labels(annotations):
    labels = [ann["category_id"] for ann in annotations]
    return labels


def random_cut(image,annotations_list,img_data,size,weights=None):
    x_max = max(0,image["width"]-size[0])
    y_max = max(0,image["height"]-size[1])
    image_info = {}
    image_info["height"] =size[1]
    image_info["width"] =size[0]
    obj_ann_bboxes = get_expand_bboxes_in_annotations(annotations_list,2)
    labels = get_labels(annotations_list)
    if len(annotations_list)==0:
        return None,None,None
    count = 1
    while count<100:
        t_bbox = odb.random_bbox_in_bboxes(obj_ann_bboxes,size,weights,labels)
        t_bbox[1] = max(0,min(t_bbox[1],y_max))
        t_bbox[0] = max(0,min(t_bbox[0],x_max))
        rect = (t_bbox[1],t_bbox[0],t_bbox[1]+t_bbox[3],t_bbox[0]+t_bbox[2])
        new_image_info,new_annotations_list,new_image_data = cut(annotations_list,img_data,rect)
        if new_annotations_list is not None and len(new_annotations_list)>0:
            return (new_image_info,new_annotations_list,new_image_data)
        ++count

    return None,None,None

def random_cutv1(image,annotations_list,img_data,size):
    res = []
    x_max = max(0,image["width"]-size[0])
    y_max = max(0,image["height"]-size[1])
    image_info = {}
    image_info["height"] =size[1]
    image_info["width"] =size[0]
    obj_ann_bboxes = get_expand_bboxes_in_annotations(annotations_list,2)
    if len(annotations_list)==0:
        return res

    for t_bbox in obj_ann_bboxes:
        t_bbox = list(t_bbox)
        t_bbox[1] = max(0,min(t_bbox[1],y_max))
        t_bbox[0] = max(0,min(t_bbox[0],x_max))
        t_bbox = odb.random_bbox_in_bbox(t_bbox,size)
        rect = (t_bbox[1],t_bbox[0],t_bbox[1]+t_bbox[3],t_bbox[0]+t_bbox[2])
        new_image_info,new_annotations_list,new_image_data = cut(annotations_list,img_data,rect)
        if new_annotations_list is not None and len(new_annotations_list)>0:
            res.append((new_image_info,new_annotations_list,new_image_data))
    return res

'''
image_data:[h,w,c]
bbox:[ymin,xmin,ymax,xmax)
'''
def cut(annotations_list,img_data,bbox,threshold=2e-4):
    bbox = list(bbox)
    bbox[0] = max(0,bbox[0])
    bbox[1] = max(0,bbox[1])
    bbox[2] = min(bbox[2],img_data.shape[0])
    bbox[3] = min(bbox[3],img_data.shape[1])

    size = (bbox[3]-bbox[1],bbox[2]-bbox[0])
    new_annotations_list = []
    image_info = {}
    image_info["height"] =size[1]
    image_info["width"] =size[0]
    area = size[1]*size[0]
    image_info["file_name"] = f"IMG_L{bbox[1]:06}_T{bbox[0]:06}_W{bbox[3]-bbox[1]:06}_H{bbox[2]-bbox[0]:06}"
    for obj_ann in annotations_list:
        cnts,bboxes = odb.cut_contourv2(obj_ann["segmentation"],bbox)
        label = obj_ann["category_id"]
        if len(cnts)>0:
            for cnt in cnts:
                t_bbox = odb.to_xyminwh(odb.bbox_of_contour(cnt))
                if t_bbox[2]*t_bbox[3]/area < threshold:
                    continue
                mask = np.zeros(shape=[size[1],size[0]],dtype=np.uint8)
                segmentation = cv.drawContours(mask,np.array([cnt]),-1,color=(1),thickness=cv.FILLED)
                new_annotations_list.append({"bbox":t_bbox,"segmentation":segmentation,"category_id":label})
    if len(new_annotations_list)>0:
        return (image_info,new_annotations_list,wmli.sub_image(img_data,bbox))
    else:
        return None,None,None

def view_data(image_file,json_file,label_text_to_id=lambda x:int(x),color_fn=None,alpha=0.4):
    image,annotation_list = read_labelme_data(json_file,label_text_to_id)
    image_data = wmli.imread(image_file)
    do_scale = False
    if image_data.shape[0]>2048 or image_data.shape[1]>2048:
        scale = min(2048.0/image_data.shape[0],2048.0/image_data.shape[1])
        size = (int(image_data.shape[1]*scale),int(image_data.shape[0]*scale))
        image_data = cv.resize(image_data,size)
        do_scale = True
    else:
        size = (image_data.shape[1],image_data.shape[0])

    for ann in annotation_list:
        if color_fn is not None:
            color = list(color_fn(ann["category_id"]))
        else:
            color = [random.random()*255, random.random()*255, random.random()*255]
        color = np.array([[color]],dtype=np.float)
        mask = ann["segmentation"]
        if do_scale:
            mask = cv.resize(mask,size)
        mask = np.expand_dims(mask,axis=-1)
        image_data  = (image_data*(np.array([[[1]]],dtype=np.float32) - mask * alpha)).astype(np.uint8) + (mask * color * alpha).astype(np.uint8)

    labels,bboxes = get_labels_and_bboxes(image,annotation_list)
    image_data = odv.bboxes_draw_on_imgv2(image_data,classes=labels,bboxes=bboxes,thickness=2)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_data)
    plt.show()

def data_statistics(data_dir):
    files = get_files(data_dir)
    statistics_data = {}
    file_statistics_data = {}

    for img_file,json_file in files:
        image, annotations_list = read_labelme_data(json_file,lambda x:x)
        temp_file_statistics_data = {}
        for ann in annotations_list:
            label = ann["category_id"]
            if label in statistics_data:
                statistics_data[label] += 1
            else:
                statistics_data[label] = 1
            temp_file_statistics_data[label] = 1
        for k in temp_file_statistics_data.keys():
            if k in file_statistics_data:
                file_statistics_data[k] += 1
            else:
                file_statistics_data[k] = 1
    if len(files)<1:
        return
    print(f"Data size {len(files)}.")
    print("Num of each classes")
    wmlu.show_dict(statistics_data)
    print("Num of each classes in files")
    wmlu.show_dict(file_statistics_data)
    _file_statistics_data = {}
    for k,v in file_statistics_data.items():
        _file_statistics_data[k] = v*100.0/len(files)


    total_num = 0
    for v in statistics_data.values():
        total_num += v

    _statistics = {}
    for k,v in statistics_data.items():
        _statistics[k] = 100.0*v/total_num

    print("Percent of each classes")
    wmlu.show_dict(_statistics)

    print("Percent of each classes in files")
    wmlu.show_dict(_file_statistics_data)


if __name__ == "__main__":
    data_statistics("/home/vghost/ai/mldata/qualitycontrol/rdatasv3")



