from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import numpy as np
from pycocotools import mask
import iotoolkit.label_map_util as label_map_util
import wml_utils as wmlu
import sys
import object_detection2.keypoints as odk

def get_yxyx_bbox(bbox_data):
    (x, y, width, height) = tuple(bbox_data)
    return y,x,y+height,x+width

def get_xy_kps(kps_data):
    kps_data = np.array(kps_data,dtype=np.float32)
    kps_data = np.reshape(kps_data,[-1,3])
    return kps_data

ID2NAMES = {
    1:"person",
    2:"lefthand",
    3:"righthand",
    4:"face",
}

class COCOWholeBodyData:
    def __init__(self,trans_label=None,include_masks=False,no_crowd=True):
        '''

        Args:
            trans_label: label fn(label) : return transed label is label is useful else return None
            include_masks:
        '''
        self.images = None
        self.annotations_index = None
        self.image_dir = None
        self.include_masks = include_masks
        self.category_index = None
        self.trans_label = trans_label
        self.no_crowd = no_crowd

    def get_image_full_path(self,image):
        filename = image['file_name']
        return os.path.join(self.image_dir, filename)

    def read_data(self,annotations_file,image_dir):
        with open(annotations_file, 'r') as fid:
            groundtruth_data = json.load(fid)
            images = groundtruth_data['images']
            category_index = label_map_util.create_category_index(
                groundtruth_data['categories'])

            annotations_index = {}
            if 'annotations' in groundtruth_data:
                print('Found groundtruth annotations. Building annotations index.')
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
            print('%d images are missing annotations.',
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
        lefthand_bboxes = []
        righthand_bboxes = []
        face_bboxes = []
        keypoints = []
        if image["file_name"] == "000000292082.jpg":
            print("A")
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
            org_category_id = category_id
            iscrowd = object_annotations['iscrowd']
            if self.no_crowd and iscrowd:
                continue
            if self.trans_label is not None:
                category_id = self.trans_label(category_id)
                if category_id is None:
                    continue

            cur_kps = []
            cur_kps.append(get_xy_kps(object_annotations["keypoints"]))
            lh_kps = get_xy_kps(object_annotations["lefthand_kpts"])
            cur_kps.append(lh_kps)
            rh_kps = get_xy_kps(object_annotations["righthand_kpts"])
            cur_kps.append(rh_kps)
            fc_kps = get_xy_kps(object_annotations['face_kpts'])
            cur_kps.append(fc_kps)
            keypoints.append(np.concatenate(cur_kps,axis=0))

            lh_box = get_yxyx_bbox(object_annotations['lefthand_box'])
            #lh_box = odk.expand_yxyx_bbox_by_kps(lh_box,lh_kps,threshold=0.1)
            lefthand_bboxes.append(lh_box)
            rh_box = get_yxyx_bbox(object_annotations['righthand_box'])
            #rh_box = odk.expand_yxyx_bbox_by_kps(rh_box,rh_kps,threshold=0.1)
            righthand_bboxes.append(rh_box)
            fc_box = get_yxyx_bbox(object_annotations['face_box'])
            #fc_box = odk.expand_yxyx_bbox_by_kps(fc_box,fc_kps,threshold=0.1)
            face_bboxes.append(fc_box)

            xmin.append(float(x) )
            xmax.append(float(x + width))
            ymin.append(float(y))
            ymax.append(float(y + height))

            is_crowd.append(iscrowd)
            category_ids.append(category_id)
            category_names.append(str(self.category_index[org_category_id]['name'].encode('utf8'),encoding='utf-8'))
            area.append(object_annotations['area'])

            if self.include_masks:
                run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                                    image_height, image_width)
                binary_mask = mask.decode(run_len_encoding)
                if not object_annotations['iscrowd']:
                    binary_mask = np.amax(binary_mask, axis=2)
                binary_masks.append(binary_mask)

        boxes = np.array(list(zip(ymin,xmin,ymax,xmax)))
        lefthand_bboxes = np.array(lefthand_bboxes)
        righthand_bboxes = np.array(righthand_bboxes)
        face_bboxes = np.array(face_bboxes)
        keypoints = np.array(keypoints)

        if len(binary_masks)>0:
            binary_masks = np.stack(binary_masks,axis=0)
        else:
            binary_masks = None

        if len(category_ids)==0:
            print("No annotation: ", full_path)
            sys.stdout.flush()
            return None,None,None,None,None,None,None,None,None
        category_ids = np.array(category_ids,dtype=np.int32)
        return full_path,[image_height,image_width],category_ids,category_names,boxes,\
               lefthand_bboxes,righthand_bboxes,face_bboxes,keypoints, \
               binary_masks,area,is_crowd,num_annotations_skipped

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        res = self.get_image_annotation(image)

    @staticmethod
    def trans2bboxes(data):
        full_path,img_shape,category_ids,category_names,boxes, \
               lefthand_bboxes,righthand_bboxes,face_bboxes,keypoints, \
               binary_masks,area,is_crowd,num_annotations_skipped = data
        assert np.all(np.array(category_ids)==1),f"Error category {category_ids}"
        lh_nr = lefthand_bboxes.shape[0]
        rh_nr = righthand_bboxes.shape[0]
        fc_nr = face_bboxes.shape[0]
        lh_category = np.ones([lh_nr],dtype=np.int32)*2
        rh_category = np.ones([rh_nr],dtype=np.int32)*3
        fc_category = np.ones([fc_nr],dtype=np.int32)*4
        all_bboxes = np.concatenate([boxes,lefthand_bboxes,righthand_bboxes,face_bboxes],axis=0)
        categorys = np.concatenate([category_ids,lh_category,rh_category,fc_category],axis=0)
        mask = np.max(all_bboxes,axis=-1)>0.5
        all_bboxes = all_bboxes[mask]
        categorys = categorys[mask]
        category_names = [ID2NAMES[id] for id in categorys]

        return full_path,img_shape,categorys,category_names,all_bboxes,keypoints


    def get_items(self):
        for image in self.images:
            res = self.get_image_annotation(image)
            if res[0] is not None:
                yield res

    def get_boxes_items(self):
        for image in self.images:
            anno_data = self.get_image_annotation(image)
            if anno_data[0] is not None:
                yield self.trans2bboxes(anno_data)

if __name__ == "__main__":
    import img_utils as wmli
    import object_detection2.visualization as odv
    import matplotlib.pyplot as plt
    data = COCOWholeBodyData()
    data.read_data("/home/wj/ai/mldata/coco/annotations/coco_wholebody_val_v1.0.json",
                   image_dir="/home/wj/ai/mldata/coco/val2017")
    save_dir = "/home/wj/ai/mldata/0day/tmp"
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)
    for i,x in enumerate(data.get_boxes_items()):
        full_path, img_shape, categorys, category_names, bboxes,kps =  x
        if 2 not in categorys and 3 not in categorys:
            continue
        img = wmli.imread(full_path)
        def text_fn(classes,scores):
            return ID2NAMES[classes]
        img = odv.draw_bboxes(
        img=img, classes=categorys, scores=None, bboxes=bboxes, color_fn = None,
            text_fn = text_fn, thickness = 4,
        show_text = True,
        font_scale = 0.8,is_relative_coordinate=False)
        #img = odv.draw_keypoints(img,kps,no_line=True)
        save_path = wmlu.change_dirname(full_path,save_dir)
        wmli.imwrite(save_path,img)
        if i>100:
            break

