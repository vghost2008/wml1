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
import sys
import cv2

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
        print(file_path)
        data_str = f.read()
        try:
            json_data = json.loads(data_str)
            img_width = int(json_data["imageWidth"])
            img_height = int(json_data["imageHeight"])
            image["height"] = int(img_height)
            image["width"] = int(img_width)
            image["file_name"] = wmlu.base_name(file_path)
            for shape in json_data["shapes"]:
                mask = np.zeros(shape=[img_height,img_width],dtype=np.uint8)
                all_points = np.array([shape["points"]]).astype(np.int32)
                if len(all_points)<1:
                    continue
                points = np.transpose(all_points[0])
                x,y = np.vsplit(points,2)
                x = np.minimum(np.maximum(0,x),img_width-1)
                y = np.minimum(np.maximum(0,y),img_height-1)
                xmin = np.min(x)
                xmax = np.max(x)
                ymin = np.min(y)
                ymax = np.max(y)
                segmentation = cv.drawContours(mask,all_points,-1,color=(1),thickness=cv.FILLED)
                if label_text_to_id is not None:
                    label = label_text_to_id(shape["label"])
                else:
                    label = shape["label"]
                annotations_list.append({"bbox":(xmin,ymin,xmax-xmin+1,ymax-ymin+1),"segmentation":segmentation,"category_id":label})
        except:
            pass
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
        ymax = ymin+t_box[3]/height
        bboxes.append([ymin,xmin,ymax,xmax])
        labels.append(ann["category_id"])
    return np.array(labels),np.array(bboxes)

def get_labels_bboxes_and_masks(image,annotations_list):
    labels = []
    bboxes = []
    masks = []
    width = image["width"]
    height = image["height"]
    for ann in annotations_list:
        t_box = ann["bbox"]
        xmin = t_box[0]/width
        ymin = t_box[1]/height
        xmax = xmin+t_box[2]/width
        ymax = ymin+t_box[3]/height
        bboxes.append([ymin,xmin,ymax,xmax])
        labels.append(ann["category_id"])
        masks.append(ann["segmentation"])
    return np.array(labels),np.array(bboxes),np.array(masks)

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

'''
获取标注box scale倍大小的扩展box(面积为scale*scale倍大)
box的中心点不变
'''
def get_expand_bboxes_in_annotations(annotations,scale=2):
   bboxes = [ann["bbox"] for ann in annotations]
   bboxes = odb.expand_bbox(bboxes,scale)
   return bboxes

'''
获取标注box 扩展为size大小的box
box的中心点不变
'''
def get_expand_bboxes_in_annotationsv2(annotations,size):
    bboxes = [ann["bbox"] for ann in annotations]
    bboxes = odb.expand_bbox_by_size(bboxes,size)
    return bboxes

def get_labels(annotations):
    labels = [ann["category_id"] for ann in annotations]
    return labels

'''
size:(h,w)
'''
def resize(image,annotations_list,img_data,size):
    res_image = copy.deepcopy(image)
    res_image["width"] = size[1]
    res_image["height"] = size[0]
    res_ann = []
    for ann in  annotations_list:
        bbox = copy.deepcopy(ann["bbox"])
        #segmentation = wmli.resize_img(ann["segmentation"],size)
        segmentation = cv2.resize(ann["segmentation"],dsize=size,interpolation=cv2.INTER_NEAREST)
        category = copy.deepcopy(ann["category_id"])
        res_ann.append({"bbox":bbox,"segmentation":segmentation,"category_id":category})
    res_img_data = wmli.resize_img(img_data,size)

    return res_image,res_ann,res_img_data

'''
从目标集中随机的选一个目标并从中截图
随机的概率可以通过weights指定
'''
def random_cut(image,annotations_list,img_data,size,weights=None,threshold=0.15):
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
        new_image_info,new_annotations_list,new_image_data = cut(annotations_list,img_data,rect,threshold=threshold)
        if new_annotations_list is not None and len(new_annotations_list)>0:
            return (new_image_info,new_annotations_list,new_image_data)
        ++count

    return None,None,None

'''
size:[H,W]
在每一个标目标附近裁剪出一个子图
'''
def random_cutv1(image,annotations_list,img_data,size,threshold=0.15):
    res = []
    x_max = max(0,image["width"]-size[0])
    y_max = max(0,image["height"]-size[1])
    image_info = {}
    image_info["height"] =size[1]
    image_info["width"] =size[0]
    obj_ann_bboxes = get_expand_bboxes_in_annotationsv2(annotations_list,size)
    if len(annotations_list)==0:
        return res

    for t_bbox in obj_ann_bboxes:
        t_bbox = list(t_bbox)
        t_bbox[1] = max(0,min(t_bbox[1],y_max))
        t_bbox[0] = max(0,min(t_bbox[0],x_max))
        t_bbox = odb.random_bbox_in_bbox(t_bbox,size)
        rect = (t_bbox[1],t_bbox[0],t_bbox[1]+t_bbox[3],t_bbox[0]+t_bbox[2])
        new_image_info,new_annotations_list,new_image_data = cut(annotations_list,img_data,rect,threshold=threshold)
        if new_annotations_list is not None and len(new_annotations_list)>0:
            res.append((new_image_info,new_annotations_list,new_image_data))
    return res


'''
ref_bbox: [N,4] relative coordinate,[ymin,xmin,ymax,xmax]
'''
def random_cutv2(image,annotations_list,ref_bbox,img_data,size,weights=None):
    x_max = max(0,image["width"]-size[0])
    y_max = max(0,image["height"]-size[1])
    image_info = {}
    image_info["height"] =size[1]
    image_info["width"] =size[0]
    obj_ann_bboxes = []
    for bbox in ref_bbox:
        ymin = int(bbox[0]*image["height"])
        xmin = int(bbox[1]*image["width"])
        width = int((bbox[3]-bbox[1])*image["width"])
        height = int((bbox[2]-bbox[0])*image["height"])
        obj_ann_bboxes.append([xmin,ymin,width,height])
    labels = get_labels(annotations_list)
    t_bbox = odb.random_bbox_in_bboxes(obj_ann_bboxes,size,weights,labels)
    t_bbox[1] = max(0,min(t_bbox[1],y_max))
    t_bbox[0] = max(0,min(t_bbox[0],x_max))
    rect = (t_bbox[1],t_bbox[0],t_bbox[1]+t_bbox[3],t_bbox[0]+t_bbox[2])
    new_image_info,new_annotations_list,new_image_data = cut(annotations_list,img_data,rect,return_none_if_no_ann=False)
    return (new_image_info,new_annotations_list,new_image_data)


'''
image_data:[h,w,c]
bbox:[ymin,xmin,ymax,xmax)
'''
def cut(annotations_list,img_data,bbox,threshold=0.15,return_none_if_no_ann=True):
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
        cnts,bboxes,ratios = odb.cut_contourv2(obj_ann["segmentation"],bbox)
        label = obj_ann["category_id"]
        if len(cnts)>0:
            for i,cnt in enumerate(cnts):
                ratio = ratios[i]
                t_bbox = odb.to_xyminwh(odb.bbox_of_contour(cnt))
                if ratio<threshold:
                    continue
                mask = np.zeros(shape=[size[1],size[0]],dtype=np.uint8)
                segmentation = cv.drawContours(mask,np.array([cnt]),-1,color=(1),thickness=cv.FILLED)
                new_annotations_list.append({"bbox":t_bbox,"segmentation":segmentation,"category_id":label})
    if (len(new_annotations_list)>0) or (not return_none_if_no_ann):
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

class LabelMeData(object):
    def __init__(self,label_text2id=None,shuffle=False):
        self.files = None
        self.label_text2id = label_text2id
        self.shuffle = shuffle
        
    def read_data(self,dir_path):
        self.files = get_files(dir_path)
        if self.shuffle:
            random.shuffle(self.files)

    def get_items(self):
        '''
        :return: 
        full_path,img_size,category_ids,category_names,boxes,binary_masks,area,is_crowd,num_annotations_skipped
        '''
        for i,(img_file, json_file) in enumerate(self.files):
            sys.stdout.write('\r>> read data %d/%d' % (i + 1, len(self.files)))
            sys.stdout.flush()
            image, annotations_list = read_labelme_data(json_file, None)
            labels_names,bboxes = get_labels_and_bboxes(image,annotations_list)
            masks = [ann["segmentation"] for ann in annotations_list]
            if len(masks)>0:
                masks = np.stack(masks,axis=0)
            
            if self.label_text2id is not None:
                labels = [self.label_text2id(x) for x in labels_names]
            else:
                labels = None
                
            yield img_file, [image['height'],image['width']],labels, labels_names, bboxes, masks, None, None,None 
    
    def get_boxes_items(self):
        '''
        :return: 
        full_path,img_size,category_ids,boxes,is_crowd
        '''
        for i,(img_file, json_file) in enumerate(self.files):
            sys.stdout.write('\r>> read data %d/%d' % (i + 1, len(self.files)))
            sys.stdout.flush()
            image, annotations_list = read_labelme_data(json_file, None)
            labels_names,bboxes = get_labels_and_bboxes(image,annotations_list)
            labels = [self.label_text2id(x) for x in labels_names]
            yield img_file,[image['height'],image['width']],labels, bboxes,  None

if __name__ == "__main__":
    #data_statistics("/home/vghost/ai/mldata/qualitycontrol/rdatasv3")
    import img_utils as wmli
    import object_detection_tools.visualization as odv
    import matplotlib.pyplot as plt
    ID_TO_TEXT = {1:{"id":1,"name":"a"},2:{"id":2,"name":"b"},3:{"id":3,"name":"c"}}
    NAME_TO_ID = {}
    for k,v in ID_TO_TEXT.items():
        NAME_TO_ID[v["name"]] = v["id"]
    def name_to_id(name):
        return NAME_TO_ID[name]

    data = LabelMeData(label_text2id=name_to_id,shuffle=True)
    data.read_data("/data/mldata/qualitycontrol/rdatasv5_splited/rdatasv5")
    data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatav8_preproc")
    for x in data.get_items():
        full_path, category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x
        img = wmli.imread(full_path)


        def text_fn(classes, scores):
            return ID_TO_TEXT[classes]['name']

        odv.draw_bboxes_and_maskv2(
            img=img, classes=category_ids, scores=None, bboxes=boxes, masks=binary_mask, color_fn=None,
            text_fn=text_fn, thickness=4,
            show_text=True,
            fontScale=0.8)
        plt.figure()
        plt.imshow(img)
        plt.show()
