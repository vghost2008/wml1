import json
import wml_utils as wmlu
import numpy as np
import os
import cv2 as cv
import sys
import random
from iotoolkit.labelme_toolkit import get_labels_and_bboxes





def get_files(dir_path, sub_dir_name):
    img_dir = os.path.join(dir_path, sub_dir_name,'images')
    label_dir = os.path.join(dir_path, sub_dir_name,"v2.0","polygons")
    res = []
    json_files = wmlu.recurse_get_filepath_in_dir(label_dir,suffix=".json")
    for jf in json_files:
        base_name = wmlu.base_name(jf)
        igf = os.path.join(img_dir, base_name + ".jpg")
        if os.path.exists(igf):
            res.append((igf, jf))
        else:
            print(f"ERROR: Find {igf} faild, json file is {jf}")

    return res


class MapillaryVistasData(object):
    def __init__(self, label_text2id=None, shuffle=False, sub_dir_name="training",ignored_labels=[],label_map={},
                 allowed_labels_fn=None):
        self.files = None
        self.label_text2id = label_text2id
        self.shuffle = shuffle
        self.sub_dir_name = sub_dir_name
        self.ignored_labels = ignored_labels
        self.label_map = label_map
        self.allowed_labels_fn = None if allowed_labels_fn is None or len(allowed_labels_fn)==0 else allowed_labels_fn
        if self.allowed_labels_fn is not None and isinstance(self.allowed_labels_fn,list):
            self.allowed_labels_fn = lambda x:x in allowed_labels_fn

    def read_data(self, dir_path):
        self.files = get_files(dir_path, self.sub_dir_name)
        if self.shuffle:
            random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def get_items(self,beg=0,end=None,filter=None):
        '''
        :return:
        binary_masks [N,H,W], value is 0 or 1,
        full_path,img_size,category_ids,category_names,boxes,binary_masks,area,is_crowd,num_annotations_skipped
        '''
        if end is None:
            end = len(self.files)
        if beg is None:
            beg = 0
        for i, (img_file, json_file) in enumerate(self.files[beg:end]):
            if filter is not None and not filter(img_file,json_file):
                continue
            print(img_file,json_file)
            sys.stdout.write('\r>> read data %d/%d' % (i + 1, len(self.files)))
            sys.stdout.flush()
            image, annotations_list = self.read_json(json_file)
            labels_names, bboxes = get_labels_and_bboxes(image, annotations_list)
            masks = [ann["segmentation"] for ann in annotations_list]
            if len(masks) > 0:
                try:
                    masks = np.stack(masks, axis=0)
                except:
                    print("ERROR: stack masks faild.")
                    masks = None

            if self.label_text2id is not None:
                labels = [self.label_text2id(x) for x in labels_names]
            else:
                labels = None

            yield img_file, [image['height'], image['width']], labels, labels_names, bboxes, masks, None, None, None

    def get_boxes_items(self):
        '''
        :return:
        full_path,img_size,category_ids,boxes,is_crowd
        '''
        for i, (img_file, json_file) in enumerate(self.files):
            sys.stdout.write('\r>> read data %d/%d' % (i + 1, len(self.files)))
            sys.stdout.flush()
            image, annotations_list = self.read_json(json_file,use_semantic=False)
            labels_names, bboxes = get_labels_and_bboxes(image, annotations_list)
            labels = [self.label_text2id(x) for x in labels_names]
            #file, img_size,category_ids, labels_text, bboxes, binary_mask, area, is_crowd, _
            yield img_file, [image['height'], image['width']], labels, labels_names,bboxes, None,None,None,None

    def read_json(self,file_path,use_semantic=True):
        annotations_list = []
        image = {}
        with open(file_path, "r", encoding="gb18030") as f:
            print(file_path)
            data_str = f.read()
            try:
                json_data = json.loads(data_str)
                img_width = int(json_data["width"])
                img_height = int(json_data["height"])
                image["height"] = int(img_height)
                image["width"] = int(img_width)
                image["file_name"] = wmlu.base_name(file_path)
                for shape in json_data["objects"]:
                    #label = shape["label"].split("--")[-1]
                    label = shape["label"]
                    if self.ignored_labels is not None and label in self.ignored_labels:
                        continue
                    if self.allowed_labels_fn is not None and not self.allowed_labels_fn(label):
                        continue
                    if self.label_map is not None and label in self.label_map:
                        label = self.label_map[label]
                    mask = np.zeros(shape=[img_height, img_width], dtype=np.uint8)
                    all_points = np.array([shape["polygon"]]).astype(np.int32)
                    if len(all_points) < 1:
                        continue
                    points = np.transpose(all_points[0])
                    x, y = np.vsplit(points, 2)
                    x = np.reshape(x, [-1])
                    y = np.reshape(y, [-1])
                    x = np.minimum(np.maximum(0, x), img_width - 1)
                    y = np.minimum(np.maximum(0, y), img_height - 1)
                    xmin = np.min(x)
                    xmax = np.max(x)
                    ymin = np.min(y)
                    ymax = np.max(y)
                    if use_semantic:
                        segmentation = cv.drawContours(mask, all_points, -1, color=(1), thickness=cv.FILLED)
                        annotations_list.append({"bbox": (xmin, ymin, xmax - xmin + 1, ymax - ymin + 1),
                                                 "segmentation": segmentation,
                                                 "category_id": label,
                                                 "points_x": x,
                                                 "points_y": y})
                    else:
                        annotations_list.append({"bbox": (xmin, ymin, xmax - xmin + 1, ymax - ymin + 1),
                                             "category_id": label,
                                             "points_x": x,
                                             "points_y": y})
            except:
                print(f"Read file {os.path.basename(file_path)} faild.")
                pass
        if use_semantic:
            '''
            Each pixel only belong to one classes, and the latter annotation will overwrite the previous
            '''
            if len(annotations_list)>2:
                mask = 1-annotations_list[-1]['segmentation']
                for i in reversed(range(len(annotations_list)-1)):
                    annotations_list[i]['segmentation'] = np.logical_and(annotations_list[i]['segmentation'], mask)
                    mask = np.logical_and(mask,1-annotations_list[i]['segmentation'])
        return image, annotations_list


if __name__ == "__main__":
    id = 0
    # data_statistics("/home/vghost/ai/mldata/qualitycontrol/rdatasv3")
    import img_utils as wmli
    import object_detection_tools.visualization as odv
    import matplotlib.pyplot as plt

    NAME2ID = {}
    ID2NAME = {}


    def name_to_id(x):
        global id
        if x in NAME2ID:
            return NAME2ID[x]
        else:
            NAME2ID[x] = id
            ID2NAME[id] = x
            id += 1
            return NAME2ID[x]


    ignored_labels = [
        'manhole', 'dashed', 'other-marking', 'static', 'front', 'back',
        'solid', 'catch-basin','utility-pole', 'pole', 'street-light','direction-back', 'direction-front'
         'ambiguous', 'other','text','diagonal','left','right','water-valve','general-single','temporary-front',
        'wheeled-slow','parking-meter','split-left-or-straight','split-right-or-straight','zigzag',
        'give-way-row','ground-animal','phone-booth','give-way-single','garage','temporary-back','caravan','other-barrier',
        'chevron','pothole','sand'
    ]
    label_map = {
        'individual':'person',
        'cyclists':'person',
        'other-rider':'person'
    }
    data = MapillaryVistasData(label_text2id=name_to_id, shuffle=False, ignored_labels=ignored_labels,label_map=label_map)
    # data.read_data("/data/mldata/qualitycontrol/rdatasv5_splited/rdatasv5")
    # data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatav10_preproc")
    # data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatasv10_neg_preproc")
    data.read_data(wmlu.home_dir("ai/mldata/mapillary_vistas/mapillary-vistas-dataset_public_v2.0"))


    def filter(x):
        return x in ['general-single', 'parking', 'temporary', 'general-horizontal']
        # return x in ['terrain']
        # return x in ['car']


    # data.read_data("/home/vghost/ai/mldata2/qualitycontrol/x")
    for x in data.get_items():
        full_path, img_info, category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x
        img = wmli.imread(full_path)


        def text_fn(classes, scores):
            return f"{ID2NAME[classes]}"


        if False:
            is_keep = [filter(x) for x in category_names]
            category_ids = np.array(category_ids)[is_keep]
            boxes = np.array(boxes)[is_keep]
            binary_mask = np.array(binary_mask)[is_keep]
        if len(category_ids) == 0:
            continue

        wmlu.show_dict(NAME2ID)
        odv.draw_bboxes_and_maskv2(
            img=img, classes=category_ids, scores=None, bboxes=boxes, masks=binary_mask, color_fn=None,
            text_fn=text_fn, thickness=4,
            show_text=True,
            fontScale=0.8)
        plt.figure()
        plt.imshow(img)
        plt.show()

