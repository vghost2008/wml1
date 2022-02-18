import json
import os.path as osp
import object_detection2.bboxes as odb
import numpy as np
import object_detection2.visualization as odv
import wml_utils as wmlu


class Object365:
    def __init__(self,absolute_coord=True):
        self.json_path = None
        self.img_dir = None
        self.data = None
        self.absolute_coord = absolute_coord
        self.images_data = {}
        self.annotation_data = {}
        self.images_id = []
        self.id2name = {}

    def read_data(self,json_path,img_dir):
        self.json_path = json_path
        self.img_dir = img_dir
        with open(self.json_path,"r") as f:
            self.data = json.load(f)
        self.images_data = {}
        self.images_id = []
        self.annotation_data = {}

        for d in self.data['images']:
            id = d['id']
            self.images_id.append(id)
            self.images_data[id] = d
        for d in self.data['annotations']:
            id = d['image_id']
            self.add_anno_data(self.annotation_data,id,d)
        name_dict = {}
        for x in self.data['categories']:
            name_dict[x['id']] = x['name']
        self.id2name = name_dict

    @staticmethod
    def add_anno_data(dict_data,id,item_data):
        if id in dict_data:
            dict_data[id].append(item_data)
        else:
            dict_data[id] = [item_data]

    def __len__(self):
        return len(self.images_id)

    def __getitem__(self, item):
        try:
            id = self.images_id[item]
            item_data = self.annotation_data[id]
            is_crowd = []
            bboxes = []
            labels = []
            labels_names = []
            for data in item_data:
                is_crowd.append(data['iscrowd'])
                bboxes.append(data['bbox'])
                label = data['category_id']
                label_text = self.id2name[label]
                labels.append(label)
                labels_names.append(label_text)
            bboxes = np.array(bboxes,dtype=np.float32)
            bboxes[...,2:] = bboxes[...,:2]+bboxes[...,2:]
            bboxes = odb.npchangexyorder(bboxes)
            is_crowd = np.array(is_crowd,dtype=np.float32)
            image_data = self.images_data[id]
            shape = [image_data['height'],image_data['width']]
            if not self.absolute_coord:
                bboxes = odb.absolutely_boxes_to_relative_boxes(bboxes,width=shape[1],height=shape[0])
            img_name = image_data['file_name']
            img_file = osp.join(self.img_dir,img_name)
            return img_file, shape, labels, labels_names, bboxes, None, None, is_crowd, None
        except Exception as e:
            print(e)
            return None

    def get_items(self):
        for i in range(len(self.images_id)):
            res = self.__getitem__(i)
            if res is None:
                continue
            yield res


if __name__ == "__main__":
    import img_utils as wmli
    import matplotlib.pyplot as plt

    save_dir = "/home/wj/ai/mldata1/Objects365/tmp"
    wmlu.create_empty_dir(save_dir)
    data = Object365(absolute_coord=False)
    data.read_data("/home/wj/ai/mldata1/Objects365/Annotations/train/train.json","/home/wj/ai/mldata1/Objects365/Images/train/train")
    MIN_IMG_SIZE = 768
    for x in data.get_items():
        full_path, shape,category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x
        if 'car' not in category_names or len(category_ids)<20:
            continue
        img = wmli.imread(full_path)
        if img.shape[0]<MIN_IMG_SIZE or img.shape[1]<MIN_IMG_SIZE:
            img = wmli.resize_img(img,[MIN_IMG_SIZE,MIN_IMG_SIZE],keep_aspect_ratio=True)

        def text_fn(classes, scores):
            return data.id2name[classes]

        img = odv.draw_bboxes(
            img=img, classes=category_ids, scores=None, bboxes=boxes, color_fn=None,
            text_fn=text_fn, thickness=2,
            show_text=True,
            font_scale=0.8)
        save_path = osp.join(save_dir,osp.basename(full_path))
        wmli.imwrite(save_path,img)
        '''plt.figure()
        plt.imshow(img)
        plt.show()'''
