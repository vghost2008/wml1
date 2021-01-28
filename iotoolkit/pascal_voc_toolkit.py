#coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import random
import os
import object_detection.npod_toolkit as npod
import math
from semantic.visualization_utils import draw_bounding_boxes_on_image_tensors
import object_detection.visualization as odv
import wml_utils
import logging
import shutil
from object_detection.metrics import coco_evaluation
from object_detection.metrics import standard_fields
import wml_utils as wmlu
import img_utils as wmli
import copy

'''
读取VOC xml文件
file_path: xml文件路径
adjust:左，上，右，下向中间的收缩像素大小
return:
shape: image size
boxes: [N,4] relative coordinate,(ymin,xmin,ymax,xmax)
'''
def read_voc_xml(file_path,adjust=None,aspect_range=None,has_probs=False,absolute_coord=False):
    tree = ET.parse(file_path)
    root = tree.getroot()

    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    if adjust is not None:
        shape[0] = shape[0] - (adjust[1]+adjust[3])
        shape[1] = shape[1] - (adjust[0]+adjust[2])

    bboxes = []
    labels_text = []
    difficult = []
    truncated = []
    probs = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        #文件中difficult用0,1表示
        if obj.find('difficult') is not None:
            dif = int(obj.find('difficult').text)
        else:
            dif = 0
        if has_probs and obj.find("prob") is not None:
            prob = float(obj.find("prob").text)
        else:
            prob = 1.0

        if obj.find('truncated') is not None:
            trun = int(obj.find('truncated').text)
        else:
            trun = 0
        bbox = obj.find('bndbox')
        box_ok = True
        if adjust is None:
            ymin,xmin,ymax,xmax = [float(bbox.find('ymin').text),
                            float(bbox.find('xmin').text),
                            float(bbox.find('ymax').text),
                            float(bbox.find('xmax').text)]
            if math.fabs(ymax-ymin)<1e-8 or math.fabs(xmax-xmin)<1e-8:
                logging.warning("zero size box({},{},{},{}), {}".format(ymin,xmin,ymax,xmax,file_path))
                continue
            else:
                if not absolute_coord:
                    box = (max(0.,ymin / shape[0]),
                           max(0.,xmin / shape[1]),
                           min(1.,ymax / shape[0]),
                           min(1.,xmax / shape[1])
                           )
                else:
                    box = (ymin,xmin,ymax,xmax)

        else:
            assert absolute_coord==False,"Error coord type"
            ymin = float(bbox.find('ymin').text)-float(adjust[1])
            xmin = float(bbox.find('xmin').text)-float(adjust[0])
            ymax = float(bbox.find('ymax').text)-float(adjust[1])
            xmax = float(bbox.find('xmax').text)-float(adjust[0])
            if math.fabs(ymax-ymin)<1e-8 or math.fabs(xmax-xmin)<1e-8:
                logging.warning("zero size box({},{},{},{}), {}".format(ymin,xmin,ymax,xmax,file_path))
                continue
            else:
                box = (max(0.,ymin / shape[0]),
                 max(0.,xmin / shape[1]),
                 min(1.,ymax / shape[0]),
                 min(1.,xmax / shape[1])
                 )
        if aspect_range is not None:
            if float(box[2] - box[0]) / (box[3] - box[1]) > aspect_range[1] or float(box[2] - box[0]) / (box[3] - box[1]) < aspect_range[0]:
                logging.warning("large aspect.")
                box_ok = False

        if not box_ok:
            logging.warning("Ignore one box")
            continue
        bboxes.append(box)
        labels_text.append(label)
        difficult.append(dif)
        truncated.append(trun)
        probs.append(prob)

    assert len(bboxes)==len(labels_text),"error size"
    assert len(bboxes)==len(difficult),"error size"
    assert len(bboxes)==len(truncated),"error size"

    if has_probs:
        return shape, np.array(bboxes), labels_text, difficult, truncated,probs
    else:
        return shape, np.array(bboxes), labels_text, difficult, truncated,1.0

def create_text_element(doc,name,value):
    if not isinstance(value,str):
        value = str(value)
    res = doc.createElement(name)
    value = doc.createTextNode(value)
    res.appendChild(value)
    return res

'''
save_path:xml文件保存路径
file_path:图像文件路径
shape:[h,w,d]
boxes:相对大小
'''
def write_voc_xml(save_path,file_path,shape, bboxes, labels_text, difficult=None, truncated=None,probs=None,is_relative_coordinate=True):

    if len(shape)==2:
        shape = list(shape)+[1]
    if difficult is None:
        difficult = ["0"] * len(labels_text)
    if truncated is None:
        truncated = ["0"] * len(labels_text)

    doc = Document()
    objectlist  = doc.createElement("annotation")
    doc.appendChild(objectlist)

    folder = doc.createElement("folder")
    #folder_value = doc.createTextNode(os.path.basename(os.path.dirname(file_path)).decode("utf-8"))
    folder_value = doc.createTextNode(os.path.basename(os.path.dirname(file_path)))
    folder.appendChild(folder_value)
    objectlist.appendChild(folder)

    filename = doc.createElement("filename")
    #filename_value = doc.createTextNode(os.path.basename(file_path).decode("utf-8"))
    filename_value = doc.createTextNode(os.path.basename(file_path))
    filename.appendChild(filename_value)
    objectlist.appendChild(filename)

    path = doc.createElement("path")
    #path_value = doc.createTextNode(file_path.decode("utf-8"))
    path_value = doc.createTextNode(file_path)
    path.appendChild(path_value)
    objectlist.appendChild(path)

    source = doc.createElement("source")
    database = doc.createElement("database")
    database_value = doc.createTextNode("Unknown")
    database.appendChild(database_value)
    source.appendChild(database)
    objectlist.appendChild(source)

    size = doc.createElement("size")
    size.appendChild(create_text_element(doc,"width",str(shape[1])))
    size.appendChild(create_text_element(doc,"height",str(shape[0])))
    size.appendChild(create_text_element(doc,"depth",str(shape[2])))
    objectlist.appendChild(size)

    objectlist.appendChild(create_text_element(doc,"segmented","0"))

    if probs is not None:
        for (box, label, dif, trun,prob) in zip(bboxes, labels_text, difficult, truncated,probs):
            object = doc.createElement("object")
            object.appendChild(create_text_element(doc, "name", str(label)))
            object.appendChild(create_text_element(doc, "pose", "Unspecified"))
            object.appendChild(create_text_element(doc, "truncated", trun))
            object.appendChild(create_text_element(doc, "difficult", dif))
            object.appendChild(create_text_element(doc, "prob", prob))
            bndbox = doc.createElement("bndbox")
            if is_relative_coordinate:
                bndbox.appendChild(create_text_element(doc, "xmin", int(box[1] * shape[1])))
                bndbox.appendChild(create_text_element(doc, "ymin", int(box[0] * shape[0])))
                bndbox.appendChild(create_text_element(doc, "xmax", int(box[3] * shape[1])))
                bndbox.appendChild(create_text_element(doc, "ymax", int(box[2] * shape[0])))
            else:
                bndbox.appendChild(create_text_element(doc, "xmin", int(box[1])))
                bndbox.appendChild(create_text_element(doc, "ymin", int(box[0])))
                bndbox.appendChild(create_text_element(doc, "xmax", int(box[3])))
                bndbox.appendChild(create_text_element(doc, "ymax", int(box[2])))
            object.appendChild(bndbox)
            objectlist.appendChild(object)
    else:
        for (box,label,dif,trun) in zip(bboxes,labels_text,difficult,truncated):
            object = doc.createElement("object")
            object.appendChild(create_text_element(doc,"name",str(label)))
            object.appendChild(create_text_element(doc,"pose","Unspecified"))
            object.appendChild(create_text_element(doc,"truncated",trun))
            object.appendChild(create_text_element(doc,"difficult",dif))
            bndbox = doc.createElement("bndbox")
            if is_relative_coordinate:
                bndbox.appendChild(create_text_element(doc,"xmin",int(box[1]*shape[1])))
                bndbox.appendChild(create_text_element(doc,"ymin",int(box[0]*shape[0])))
                bndbox.appendChild(create_text_element(doc,"xmax",int(box[3]*shape[1])))
                bndbox.appendChild(create_text_element(doc,"ymax",int(box[2]*shape[0])))
            else:
                bndbox.appendChild(create_text_element(doc, "xmin", int(box[1])))
                bndbox.appendChild(create_text_element(doc, "ymin", int(box[0])))
                bndbox.appendChild(create_text_element(doc, "xmax", int(box[3])))
                bndbox.appendChild(create_text_element(doc, "ymax", int(box[2])))
            object.appendChild(bndbox)
            objectlist.appendChild(object)

    with open(save_path,'w') as f:
        #f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
        f.write(doc.toprettyxml(indent='\t'))

'''
file_path:图像文件路径
save_path: xml path or None
bboxes:相对坐标
'''
def writeVOCXml(file_path,bboxes, labels, save_path=None,difficult=None, truncated=None,probs=None,img_shape=None,is_relative_coordinate=True):
    if isinstance(bboxes,np.ndarray):
        bboxes = bboxes.tolist()
    if isinstance(labels,np.ndarray):
        labels = labels.tolist()
    if isinstance(difficult,np.ndarray):
        difficult = difficult.tolist()
    if isinstance(truncated, np.ndarray):
        truncated = truncated.tolist()

    if img_shape is None:
        img = mpimg.imread(file_path)
        img_shape = img.shape

    if save_path is None:
        dir_path  = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        base_name = base_name[:-4]+".xml"
        save_path = os.path.join(dir_path,base_name)

    write_voc_xml(save_path,file_path,img_shape,bboxes,labels,difficult,truncated,probs=probs,
                  is_relative_coordinate=is_relative_coordinate)

'''
与上一个版本的区别为 img shape 为输入值，不需要读图获取
file_path: image path
save_path: xml path or None
'''
def writeVOCXmlV2(file_path,shape,bboxes, labels, save_path=None,difficult=None, truncated=None,probs=None,
                  is_relative_coordinate=True):
    if isinstance(bboxes,np.ndarray):
        bboxes = bboxes.tolist()
    if isinstance(labels,np.ndarray):
        labels = labels.tolist()
    if isinstance(difficult,np.ndarray):
        difficult = difficult.tolist()
    if isinstance(truncated, np.ndarray):
        truncated = truncated.tolist()

    if save_path is None:
        dir_path  = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        base_name = base_name[:-4]+".xml"
        save_path = os.path.join(dir_path,base_name)

    write_voc_xml(save_path,file_path,shape,bboxes,labels,difficult,truncated,probs=probs,
                  is_relative_coordinate=is_relative_coordinate)

'''
file_path:图像文件路径
bboxes:相对坐标
'''
def writeVOCXmlByImg(img,img_save_path,bboxes, labels, difficult=None, truncated=None,probs=None,is_relative_coordinate=True):
    if isinstance(bboxes,np.ndarray):
        bboxes = bboxes.tolist()
    if isinstance(labels,np.ndarray):
        labels = labels.tolist()
    if isinstance(difficult,np.ndarray):
        difficult = difficult.tolist()
    if isinstance(truncated, np.ndarray):
        truncated = truncated.tolist()

    img_shape = img.shape

    dir_path  = os.path.dirname(img_save_path)
    base_name = os.path.basename(img_save_path)
    base_name = base_name[:-4]+".xml"
    save_path = os.path.join(dir_path,base_name)
    wmli.imwrite(img_save_path,img)
    write_voc_xml(save_path,img_save_path,img_shape,bboxes,labels,difficult,truncated,probs=probs,
                  is_relative_coordinate=is_relative_coordinate)

'''
return:[(image_file0,xml_file0),(image_file1,xml_file1),...]
'''
def getVOCFiles(dir_path,image_sub_dir="JPEGImages",xml_sub_dir="Annotations",img_suffix=".jpg",shuffe=False,auto_sub_dir=False,silent=False):
    if auto_sub_dir:
        jpeg_dir = os.path.join(dir_path,"JPEGImages")
        if not os.path.exists(jpeg_dir):
            jpeg_dir = dir_path
        xml_dir = os.path.join(dir_path,"Annotations")
        if not os.path.exists(xml_dir):
            xml_dir = dir_path
    else:
        if image_sub_dir is not None:
            jpeg_dir = os.path.join(dir_path,image_sub_dir)
        else:
            jpeg_dir = dir_path
        if xml_sub_dir is not None:
            xml_dir = os.path.join(dir_path,xml_sub_dir)
        else:
            xml_dir = dir_path
    inputfilenames = wml_utils.recurse_get_filepath_in_dir(jpeg_dir,suffix=img_suffix)

    img_file_paths = []
    xml_file_paths = []
    for file in inputfilenames:
        base_name = os.path.basename(file)[:-4]+".xml"
        if xml_sub_dir is not None:
            xml_path = os.path.join(xml_dir,base_name)
        else:
            xml_path = os.path.join(os.path.dirname(file),base_name)
        if os.path.exists(xml_path):
            img_file_paths.append(file)
            xml_file_paths.append(xml_path)
        elif not silent:
            print("ERROR, xml file dosen't exists: ",file,xml_path)

    res = []
    for x in zip(img_file_paths,xml_file_paths):
        res.append(list(x))
    if shuffe:
        random.shuffle(res)
    return res

def getVOCFilesV2(dir_path):
    img_files = wmlu.recurse_get_filepath_in_dir(dir_path,".jpg")
    res = []
    for f in img_files:
        xml_path = wmlu.change_suffix(f,"xml")
        if os.path.exists(xml_path) and os.path.exists(f):
            res.append([f,xml_path])
    return res

def filterVOCFilesByName(voc_files,file_names):
    res = []
    for img_file,xml_file in voc_files:
        base_name = wmlu.base_name(img_file)
        if base_name not in file_names:
            continue
        res.append((img_file,xml_file))
    return res
'''
return:[(image_file0,xml_file0),(image_file1,xml_file1),...]
'''
def removeUnmatchVOCFiles(dir_path,image_sub_dir="JPEGImages",xml_sub_dir="Annotations",img_suffix=".jpg",shuffe=False):
    if image_sub_dir is not None:
        jpeg_dir = os.path.join(dir_path,image_sub_dir)
    else:
        jpeg_dir = dir_path
    if xml_sub_dir is not None:
        xml_dir = os.path.join(dir_path,xml_sub_dir)
    else:
        xml_dir = dir_path
    inputfilenames = wml_utils.recurse_get_filepath_in_dir(jpeg_dir,suffix=img_suffix)

    good_xml_names=[]
    for file in inputfilenames:
        base_name = os.path.basename(file)[:-4]+".xml"
        xml_path = os.path.join(xml_dir,base_name)
        if os.path.exists(xml_path):
            good_xml_names.append(base_name)
        else:
            print(f"remove {file}")
            os.remove(file)

    for file in wml_utils.recurse_get_filepath_in_dir(xml_dir,suffix="xml"):
        base_name = os.path.basename(file)
        if base_name not in good_xml_names:
            print(f"remove {file}")
            os.remove(file)

class PascalVOCData(object):
    def __init__(self, label_text2id=None, shuffle=False,image_sub_dir=None,xml_sub_dir=None,has_probs=False,absolute_coord=False):
        '''

        :param label_text2id: trans a single label text to id
        :param shuffle:
        :param image_sub_dir:
        :param xml_sub_dir:
        '''
        self.files = None
        self.label_text2id = label_text2id
        self.shuffle = shuffle
        self.xml_sub_dir = xml_sub_dir
        self.image_sub_dir = image_sub_dir
        self.has_probs = has_probs
        self.absolute_coord = absolute_coord

    def read_data(self,dir_path,silent=False,img_suffix=".jpg"):
        print(f"Read {dir_path}")
        if not os.path.exists(dir_path):
            print(f"Data path {dir_path} not exists.")
            return False
        self.files = getVOCFiles(dir_path,image_sub_dir=self.image_sub_dir,
                                 xml_sub_dir=self.xml_sub_dir,
                                 img_suffix=img_suffix,
                                 silent=silent)
        if len(self.files) == 0:
            return False
            
        if self.shuffle:
            random.shuffle(self.files)
        return True

    def get_items(self):
        '''
        :return: 
        full_path,img_size,category_ids,category_names,boxes,binary_masks,area,is_crowd,num_annotations_skipped
        '''
        for img_file, xml_file in self.files:
            #print(xml_file)
            try:
                shape, bboxes, labels_names, difficult, truncated,probs = read_voc_xml(xml_file,
                                                                            adjust=None,
                                                                            aspect_range=None,
                                                                            has_probs=self.has_probs,
                                                                            absolute_coord=self.absolute_coord)

                if self.label_text2id is not None:
                    labels = [self.label_text2id(x) for x in labels_names]
                else:
                    labels = None
            except Exception as e:
                print(f"Read {xml_file} {e} faild.")
                continue
            #使用difficult表示is_crowd
            yield img_file, shape[:2],labels, labels_names, bboxes, None, None, difficult, probs

    def get_boxes_items(self):
        '''
        :return: 
        full_path,img_size,category_ids,boxes,is_crowd
        '''
        for img_file, xml_file in self.files:
            shape, bboxes, labels_names, difficult, truncated = read_voc_xml(xml_file,
                                                                             adjust=None,
                                                                             aspect_range=None,
                                                                             has_probs=False)
            labels = [self.label_text2id(x) for x in labels_names]
            #使用difficult表示is_crowd
            yield img_file,shape[:2], labels, bboxes, difficult

if __name__ == "__main__":
    #data_statistics("/home/vghost/ai/mldata/qualitycontrol/rdatasv3")
    import img_utils as wmli
    import object_detection_tools.visualization as odv
    import matplotlib.pyplot as plt

    text = []
    for i in range(ord('a'), ord('z') + 1):
        text.append(chr(i))
    for i in range(ord('A'), ord('Z') + 1):
        text.append(chr(i))
    for i in range(ord('0'), ord('9') + 1):
        text.append(chr(i))
    '''
    0:53
    1:54
    ...
    9:62
    '''
    text.append('/')
    text.append('\\')
    text.append('-')
    text.append('+')
    text.append(":")
    text.append("WORD")
    text.append("WD0")  # up
    text.append("WD3")  # right
    text.append("WD1")  # down
    text.append("WD2")  # left
    text.append("##")  # left

    text_to_id = {}
    id_to_text = {}

    for i, t in enumerate(text):
        text_to_id[t] = i + 1
        id_to_text[i + 1] = t

    text_to_id[" "] = 69

    def name_to_id(name):
        return text_to_id[name]

    data = PascalVOCData(label_text2id=name_to_id,shuffle=True)
    data.read_data("/home/vghost/ai/mldata2/ocrdata/rdatasv20/train")
    MIN_IMG_SIZE = 768
    for x in data.get_items():
        full_path, category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x
        img = wmli.imread(full_path)
        if img.shape[0]<MIN_IMG_SIZE or img.shape[1]<MIN_IMG_SIZE:
            img = wmli.resize_img(img,[MIN_IMG_SIZE,MIN_IMG_SIZE],keep_aspect_ratio=True)


        def text_fn(classes, scores):
            return id_to_text[classes]

        odv.bboxes_draw_on_imgv2(
            img=img, classes=category_ids, scores=None, bboxes=boxes, color_fn=None,
            text_fn=text_fn, thickness=2,
            show_text=True,
            fontScale=0.8)
        plt.figure()
        plt.imshow(img)
        plt.show()
