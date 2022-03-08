import json

import cv2

import wml_utils as wmlu
import numpy as np
import os
import cv2 as cv
import sys
import random
from iotoolkit.labelme_toolkit import get_labels_and_bboxes


def get_files(dir_path, img_sub_dir=None,label_sub_dir=None):
    if img_sub_dir is not None:
        img_dir = os.path.join(dir_path, img_sub_dir)
    else:
        img_dir = dir_path

    if label_sub_dir is not None:
        label_dir = os.path.join(dir_path, label_sub_dir)
    else:
        label_dir = dir_path
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


class BaiDuMaskData(object):
    def __init__(self, trans_label=None,shuffle=False, img_sub_dir=None,label_sub_dir=None,ignored_labels=[],label_map={},
                 allowed_labels_fn=None,overlap=True):
        self.files = None
        self.shuffle = shuffle
        self.trans_label = trans_label
        self.img_sub_dir = img_sub_dir
        self.label_sub_dir = label_sub_dir
        self.ignored_labels = ignored_labels
        self.label_map = label_map
        self.allowed_labels_fn = None if allowed_labels_fn is None or (isinstance(allowed_labels_fn,list ) and len(allowed_labels_fn)==0) else allowed_labels_fn
        self.overlap = overlap
        if self.allowed_labels_fn is not None and isinstance(self.allowed_labels_fn,list):
            self.allowed_labels_fn = lambda x:x in allowed_labels_fn

    def read_data(self, dir_path):
        self.files = get_files(dir_path, self.img_sub_dir,self.label_sub_dir)
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
            image, annotations_list = self.read_json(json_file,img_file)
            masks = [ann["segmentation"] for ann in annotations_list]
            labels = [ann['category_id'] for ann in annotations_list]
            labels_names = [ann['category_name'] for ann in annotations_list]
            bboxes = [ann['bbox'] for ann in annotations_list]
            if len(masks) > 0:
                try:
                    masks = np.stack(masks, axis=0)
                except:
                    print("ERROR: stack masks faild.")
                    masks = None

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
            labels = [ann['category_id'] for ann in annotations_list]
            labels_names = [ann['category_name'] for ann in annotations_list]
            bboxes = [ann['bbox'] for ann in annotations_list]
            #file, img_size,category_ids, labels_text, bboxes, binary_mask, area, is_crowd, _
            yield img_file, [image['height'], image['width']], labels, labels_names,bboxes, None,None,None,None
    @staticmethod
    def write_json(save_path,mask,labels2name,labels2color,label_trans=None,epsilon=None):
        shapes = []
        for k,n in labels2name.items():
            mask_tmp = (mask==k).astype(np.uint8)
            if np.sum(mask_tmp)==0:
                continue
            contours, hierarchy = cv.findContours(mask_tmp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            for cont in contours:
                if epsilon is not None:
                    cont = cv2.approxPolyDP(cont,epsilon,True)
                points = cont
                if len(cont.shape) == 3 and cont.shape[1] == 1:
                    points = np.squeeze(points, axis=1)
                points = points.tolist()
                if len(points)<3:
                    continue
                tmp_data = {'name':n}
                if label_trans is not None:
                    tmp_data['labelIdx'] = label_trans[k]
                else:
                    tmp_data['labelIdx'] = k
                tmp_data["points"] = points
                tmp_data['color'] = labels2color[k]
                shapes.append(tmp_data)
        with open(save_path,"w") as f:
            json.dump(shapes,f)

    def read_json(self,file_path,img_path,use_semantic=True):
        annotations_list = []
        image = {}
        img = cv.imread(img_path)
        with open(file_path, "r", encoding="gb18030") as f:
            print(file_path)
            data_str = f.read()
            try:
                json_data = json.loads(data_str)
                img_width = int(img.shape[1])
                img_height = int(img.shape[0])
                image["height"] = int(img_height)
                image["width"] = int(img_width)
                image["file_name"] = wmlu.base_name(file_path)
                for shape in json_data:
                    label = shape["labelIdx"]
                    label_name = shape['name']
                    if self.ignored_labels is not None and label in self.ignored_labels:
                        continue
                    if self.allowed_labels_fn is not None and not self.allowed_labels_fn(label):
                        continue
                    if self.label_map is not None and label in self.label_map:
                        label = self.label_map[label]
                    mask = np.zeros(shape=[img_height, img_width], dtype=np.uint8)
                    all_points = np.array([shape["points"]]).astype(np.int32)
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
                                                 "category_name": label_name,
                                                 "points_x": x,
                                                 "points_y": y})
                    else:
                        annotations_list.append({"bbox": (xmin, ymin, xmax - xmin + 1, ymax - ymin + 1),
                                             "category_id": label,
                                             "category_name": label_name,
                                             "points_x": x,
                                             "points_y": y})
            except:
                print(f"Read file {os.path.basename(file_path)} faild.")
                pass
        if use_semantic and self.overlap:
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

    ID2NAME = {1:"person",2:"seat_belt"}


    data = BaiDuMaskData( label_sub_dir="label",shuffle=False)
    data.read_data("/home/wj/ai/mldata1/safety_belt/train_1")

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

        if len(category_ids) == 0:
            continue

        odv.draw_bboxes_and_maskv2(
            img=img, classes=category_ids, scores=None, bboxes=boxes, masks=binary_mask, color_fn=None,
            text_fn=text_fn, thickness=4,
            show_text=True,
            fontScale=0.8)
        plt.figure()
        plt.imshow(img)
        plt.show()

