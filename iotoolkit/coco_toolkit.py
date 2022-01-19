from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import numpy as np
from pycocotools import mask
import tensorflow as tf
import iotoolkit.label_map_util as label_map_util
import sys

COCO_KP_NR = 17
ID_TO_COMPRESSED_ID = {}
COMPRESSED_ID_TO_ID = {}

ID_TO_TEXT={1: {u'supercategory': u'person', u'id': 1, u'name': u'person'},
            2: {u'supercategory': u'vehicle', u'id': 2, u'name': u'bicycle'},
            3: {u'supercategory': u'vehicle', u'id': 3, u'name': u'car'},
            4: {u'supercategory': u'vehicle', u'id': 4, u'name': u'motorcycle'},
            5: {u'supercategory': u'vehicle', u'id': 5, u'name': u'airplane'},
            6: {u'supercategory': u'vehicle', u'id': 6, u'name': u'bus'},
            7: {u'supercategory': u'vehicle', u'id': 7, u'name': u'train'},
            8: {u'supercategory': u'vehicle', u'id': 8, u'name': u'truck'},
            9: {u'supercategory': u'vehicle', u'id': 9, u'name': u'boat'},
            10: {u'supercategory': u'outdoor', u'id': 10, u'name': u'traffic light'},
            11: {u'supercategory': u'outdoor', u'id': 11, u'name': u'fire hydrant'},
            13: {u'supercategory': u'outdoor', u'id': 13, u'name': u'stop sign'},
            14: {u'supercategory': u'outdoor', u'id': 14, u'name': u'parking meter'},
            15: {u'supercategory': u'outdoor', u'id': 15, u'name': u'bench'},
            16: {u'supercategory': u'animal', u'id': 16, u'name': u'bird'},
            17: {u'supercategory': u'animal', u'id': 17, u'name': u'cat'},
            18: {u'supercategory': u'animal', u'id': 18, u'name': u'dog'},
            19: {u'supercategory': u'animal', u'id': 19, u'name': u'horse'},
            20: {u'supercategory': u'animal', u'id': 20, u'name': u'sheep'},
            21: {u'supercategory': u'animal', u'id': 21, u'name': u'cow'},
            22: {u'supercategory': u'animal', u'id': 22, u'name': u'elephant'},
            23: {u'supercategory': u'animal', u'id': 23, u'name': u'bear'},
            24: {u'supercategory': u'animal', u'id': 24, u'name': u'zebra'},
            25: {u'supercategory': u'animal', u'id': 25, u'name': u'giraffe'},
            27: {u'supercategory': u'accessory', u'id': 27, u'name': u'backpack'},
            28: {u'supercategory': u'accessory', u'id': 28, u'name': u'umbrella'},
            31: {u'supercategory': u'accessory', u'id': 31, u'name': u'handbag'},
            32: {u'supercategory': u'accessory', u'id': 32, u'name': u'tie'},
            33: {u'supercategory': u'accessory', u'id': 33, u'name': u'suitcase'},
            34: {u'supercategory': u'sports', u'id': 34, u'name': u'frisbee'},
            35: {u'supercategory': u'sports', u'id': 35, u'name': u'skis'},
            36: {u'supercategory': u'sports', u'id': 36, u'name': u'snowboard'},
            37: {u'supercategory': u'sports', u'id': 37, u'name': u'sports ball'},
            38: {u'supercategory': u'sports', u'id': 38, u'name': u'kite'},
            39: {u'supercategory': u'sports', u'id': 39, u'name': u'baseball bat'},
            40: {u'supercategory': u'sports', u'id': 40, u'name': u'baseball glove'},
            41: {u'supercategory': u'sports', u'id': 41, u'name': u'skateboard'},
            42: {u'supercategory': u'sports', u'id': 42, u'name': u'surfboard'},
            43: {u'supercategory': u'sports', u'id': 43, u'name': u'tennis racket'},
            44: {u'supercategory': u'kitchen', u'id': 44, u'name': u'bottle'},
            46: {u'supercategory': u'kitchen', u'id': 46, u'name': u'wine glass'},
            47: {u'supercategory': u'kitchen', u'id': 47, u'name': u'cup'},
            48: {u'supercategory': u'kitchen', u'id': 48, u'name': u'fork'},
            49: {u'supercategory': u'kitchen', u'id': 49, u'name': u'knife'},
            50: {u'supercategory': u'kitchen', u'id': 50, u'name': u'spoon'},
            51: {u'supercategory': u'kitchen', u'id': 51, u'name': u'bowl'},
            52: {u'supercategory': u'food', u'id': 52, u'name': u'banana'},
            53: {u'supercategory': u'food', u'id': 53, u'name': u'apple'},
            54: {u'supercategory': u'food', u'id': 54, u'name': u'sandwich'},
            55: {u'supercategory': u'food', u'id': 55, u'name': u'orange'},
            56: {u'supercategory': u'food', u'id': 56, u'name': u'broccoli'},
            57: {u'supercategory': u'food', u'id': 57, u'name': u'carrot'},
            58: {u'supercategory': u'food', u'id': 58, u'name': u'hot dog'},
            59: {u'supercategory': u'food', u'id': 59, u'name': u'pizza'},
            60: {u'supercategory': u'food', u'id': 60, u'name': u'donut'},
            61: {u'supercategory': u'food', u'id': 61, u'name': u'cake'},
            62: {u'supercategory': u'furniture', u'id': 62, u'name': u'chair'},
            63: {u'supercategory': u'furniture', u'id': 63, u'name': u'couch'},
            64: {u'supercategory': u'furniture', u'id': 64, u'name': u'potted plant'},
            65: {u'supercategory': u'furniture', u'id': 65, u'name': u'bed'},
            67: {u'supercategory': u'furniture', u'id': 67, u'name': u'dining table'},
            70: {u'supercategory': u'furniture', u'id': 70, u'name': u'toilet'},
            72: {u'supercategory': u'electronic', u'id': 72, u'name': u'tv'},
            73: {u'supercategory': u'electronic', u'id': 73, u'name': u'laptop'},
            74: {u'supercategory': u'electronic', u'id': 74, u'name': u'mouse'},
            75: {u'supercategory': u'electronic', u'id': 75, u'name': u'remote'},
            76: {u'supercategory': u'electronic', u'id': 76, u'name': u'keyboard'},
            77: {u'supercategory': u'electronic', u'id': 77, u'name': u'cell phone'},
            78: {u'supercategory': u'appliance', u'id': 78, u'name': u'microwave'},
            79: {u'supercategory': u'appliance', u'id': 79, u'name': u'oven'},
            80: {u'supercategory': u'appliance', u'id': 80, u'name': u'toaster'},
            81: {u'supercategory': u'appliance', u'id': 81, u'name': u'sink'},
            82: {u'supercategory': u'appliance', u'id': 82, u'name': u'refrigerator'},
            84: {u'supercategory': u'indoor', u'id': 84, u'name': u'book'},
            85: {u'supercategory': u'indoor', u'id': 85, u'name': u'clock'},
            86: {u'supercategory': u'indoor', u'id': 86, u'name': u'vase'},
            87: {u'supercategory': u'indoor', u'id': 87, u'name': u'scissors'},
            88: {u'supercategory': u'indoor', u'id': 88, u'name': u'teddy bear'},
            89: {u'supercategory': u'indoor', u'id': 89, u'name': u'hair drier'},
            90: {u'supercategory': u'indoor', u'id': 90, u'name': u'toothbrush'}}
COMPRESSED_ID_TO_TEXT = {}



KEYPOINTS_NAME = ["nose", #0
                "left_eye", #1
                "right_eye",#2
                "left_ear",#3
                "right_ear",#4
                "left_shoulder",#5
                "right_shoulder",#6
                "left_elbow",#7
                "right_elbow",#8
                "left_wrist",#9
                "right_wrist",#10
                "left_hip",#11
                "right_hip",#12
                "left_knee",#13
                "right_knee",#14
                "left_ankle",#15
                "right_ankle"]#16
JOINTS_PAIR = [[0 , 1], [1 , 2], [2 , 0], [1 , 3], [2 , 4], [3 , 5], [4 , 6], [5 , 6], [5 , 11],
[6 , 12], [11 , 12], [5 , 7], [7 , 9], [6 , 8], [8 , 10], [11 , 13], [13 , 15], [12 , 14], [14 , 16]]
j = 1
for i in range(1,81):
    while j not in ID_TO_TEXT:
        j += 1
    ID_TO_COMPRESSED_ID[j] = i
    COMPRESSED_ID_TO_ID[i] = j
    j += 1

for k,v in COMPRESSED_ID_TO_ID.items():
    COMPRESSED_ID_TO_TEXT[k] = ID_TO_TEXT[v]['name']

COCO_CLASSES_FREQ = {
"person": 30.52,
"car": 5.10,
"chair": 4.48,
"book": 2.87,
"bottle": 2.83,
"cup": 2.40,
"dining table": 1.83,
"bowl": 1.67,
"traffic light": 1.50,
"handbag": 1.44,
"umbrella": 1.33,
"bird": 1.26,
"boat": 1.25,
"truck": 1.16,
"bench": 1.14,
"sheep": 1.11,
"banana": 1.10,
"kite": 1.06,
"motorcycle": 1.01,
"backpack": 1.01,
"potted plant": 1.01,
"cow": 0.95,
"wine glass": 0.92,
"carrot": 0.91,
"knife": 0.90,
"broccoli": 0.85,
"donut": 0.83,
"bicycle": 0.83,
"skis": 0.77,
"vase": 0.77,
"horse": 0.77,
"tie": 0.76,
"cell phone": 0.75,
"orange": 0.74,
"cake": 0.74,
"sports ball": 0.74,
"clock": 0.74,
"suitcase": 0.72,
"spoon": 0.72,
"surfboard": 0.71,
"bus": 0.71,
"apple": 0.68,
"pizza": 0.68,
"tv": 0.68,
"couch": 0.67,
"remote": 0.66,
"sink": 0.65,
"skateboard": 0.64,
"elephant": 0.64,
"dog": 0.64,
"fork": 0.64,
"zebra": 0.62,
"airplane": 0.60,
"giraffe": 0.60,
"laptop": 0.58,
"tennis racket": 0.56,
"teddy bear": 0.56,
"cat": 0.55,
"train": 0.53,
"sandwich": 0.51,
"bed": 0.49,
"toilet": 0.48,
"baseball glove": 0.44,
"oven": 0.39,
"baseball bat": 0.38,
"hot dog": 0.34,
"keyboard": 0.33,
"snowboard": 0.31,
"frisbee": 0.31,
"refrigerator": 0.31,
"mouse": 0.26,
"stop sign": 0.23,
"toothbrush": 0.23,
"fire hydrant": 0.22,
"microwave": 0.19,
"scissors": 0.17,
"bear": 0.15,
"parking meter": 0.15,
"toaster": 0.03,
"hair drier": 0.02
}
ID_TO_FREQ = np.ones(shape=[91],dtype=np.float32)
for i in range(1,91):
    if i in ID_TO_TEXT:
        name = ID_TO_TEXT[i]['name']
        if name not in COCO_CLASSES_FREQ:
            print(f"Error {i}/{name}")
            raise ValueError(name)
        else:
            ID_TO_FREQ[i] = COCO_CLASSES_FREQ[name]
COMPRESSED_ID_TO_FREQ = np.ones(shape=[81],dtype=np.float32)
for i in range(1,81):
    id = COMPRESSED_ID_TO_ID[i]
    COMPRESSED_ID_TO_FREQ[i] = ID_TO_FREQ[id]

class COCOData:
    def __init__(self,filter=None,include_masks=True):
        self.images = None
        self.annotations_index = None
        self.image_dir = None
        self.include_masks = include_masks
        self.category_index = None
        self.filter=filter

    def get_image_full_path(self,image):
        filename = image['file_name']
        return os.path.join(self.image_dir, filename)

    def read_data(self,annotations_file,image_dir):
        with tf.gfile.GFile(annotations_file, 'r') as fid:
            groundtruth_data = json.load(fid)
            images = groundtruth_data['images']
            if self.filter is not None:
                images = list(filter(self.filter,images))
            category_index = label_map_util.create_category_index(
                groundtruth_data['categories'])

            annotations_index = {}
            if 'annotations' in groundtruth_data:
                tf.logging.info(
                    'Found groundtruth annotations. Building annotations index.')
                for annotation in groundtruth_data['annotations']:
                    image_id = annotation['image_id']
                    if image_id not in annotations_index:
                        annotations_index[image_id] = []
                    annotations_index[image_id].append(annotation)
            missing_annotation_count = 0
            for image in images:
                image_id = image['id']
                if image_id not in annotations_index:
                    missing_annotation_count += 1
                    annotations_index[image_id] = []
            tf.logging.info('%d images are missing annotations.',
                            missing_annotation_count)

        self.images = images
        self.annotations_index = annotations_index
        self.image_dir = image_dir
        self.category_index = category_index

    def get_image_annotation(self,image):
        image_height = image['height']
        image_width = image['width']
        image_id = image['id']

        full_path = self.get_image_full_path(image)

        xmin = []
        xmax = []
        ymin = []
        ymax = []
        is_crowd = []
        category_names = []
        category_ids = []
        area = []
        num_annotations_skipped = 0
        annotations_list = self.annotations_index[image_id]
        binary_masks = []
        for object_annotations in annotations_list:
            (x, y, width, height) = tuple(object_annotations['bbox'])
            if width <= 0 or height <= 0:
                num_annotations_skipped += 1
                continue
            if x<0 or x>=image_width  or y<0 or y>=image_height:
                num_annotations_skipped += 1
                continue
            if x + width > image_width:
                width = image_width-x
            if y + height > image_height:
                height = image_height-y

            category_id = int(object_annotations['category_id'])

            xmin.append(float(x) / image_width)
            xmax.append(float(x + width) / image_width)
            ymin.append(float(y) / image_height)
            ymax.append(float(y + height) / image_height)

            is_crowd.append(object_annotations['iscrowd'])
            category_ids.append(category_id)
            category_names.append(str(self.category_index[category_id]['name'].encode('utf8'),encoding='utf-8'))
            area.append(object_annotations['area'])

            if self.include_masks:
                run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                                    image_height, image_width)
                binary_mask = mask.decode(run_len_encoding)
                if not object_annotations['iscrowd']:
                    binary_mask = np.amax(binary_mask, axis=2)
                binary_masks.append(binary_mask)

        boxes = np.array(list(zip(ymin,xmin,ymax,xmax)))

        if len(binary_masks)>0:
            binary_masks = np.stack(binary_masks,axis=0)
        else:
            binary_masks = None

        if len(category_ids)==0:
            print("No annotation: ", full_path)
            sys.stdout.flush()
            return None,None,None,None,None,None,None,None,None
        return full_path,[image_height,image_height],category_ids,category_names,boxes,binary_masks,area,is_crowd,num_annotations_skipped

    def get_items(self):
        for image in self.images:
            res = self.get_image_annotation(image)
            if res[0] is not None:
                yield res

    def get_boxes_items(self):
        for image in self.images:
            full_path,img_size,category_ids,category_names,boxes,binary_mask,area,is_crowd,num_annotations_skipped = \
            self.get_image_annotation(image)
            if full_path is not None:
                yield full_path,img_size,category_ids,boxes,is_crowd

if __name__ == "__main__":
    import img_utils as wmli
    import object_detection_tools.visualization as odv
    import matplotlib.pyplot as plt
    data = COCOData()
    data.read_data("/data/mldata/coco/annotations/instances_train2014.json",image_dir="/data/mldata/coco/train2014")
    for x in data.get_items():
        full_path, category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x
        img = wmli.imread(full_path)
        def text_fn(classes,scores):
            return ID_TO_TEXT[classes]['name']
        odv.draw_bboxes_and_maskv2(
        img=img, classes=category_ids, scores=None, bboxes=boxes, masks=binary_mask, color_fn = None, text_fn = text_fn, thickness = 4,
        show_text = True,
        fontScale = 0.8)
        plt.figure()
        plt.imshow(img)
        plt.show()
