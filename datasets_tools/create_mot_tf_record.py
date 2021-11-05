from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import shutil
import hashlib
import io
import json
import os
import numpy as np
import PIL.Image
import wml_utils as wmlu
import img_utils as wmli
from pycocotools import mask
import tensorflow as tf

import iotoolkit.dataset_util as dataset_util
import iotoolkit.label_map_util as label_map_util
from iotoolkit.coco_toolkit import *

RECORD_IMG_SIZE = (1080//2,1920//2)

flags = tf.app.flags
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')
FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

src_file_index = 0
IMAGE_PER_RECORD = 10000

class MOTDatasets(object):
    def __init__(self,dirs):
        self.dirs = dirs
        self.tid_curr = 0
        self.dir_curr = 0

    def get_data_items(self):
        for seq_root in self.dirs:
            seqs = wmlu.get_subdir_in_dir(seq_root)
            for seq in seqs:
                seq_info = open(os.path.join(seq_root, seq, 'seqinfo.ini')).read()
                seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
                seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

                gt_txt = os.path.join(seq_root, seq, 'gt', 'gt.txt')
                if not os.path.exists(gt_txt):
                    print(f"{gt_txt} not exists!")
                    continue
                gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
                idx = np.lexsort(gt.T[:2, :])
                gt = gt[idx, :]

                fid_datas = {}
                for v in gt:
                    fid, tid, x, y, w, h, mark, label, _ = v[:9]
                    if mark == 0:
                        continue
                    if 'MOT15' not in seq_root and label !=1:
                        continue
                    if fid in fid_datas:
                        fid_datas[fid].append(v[:9])
                    else:
                        fid_datas[fid] = [v[:9]]
                tid_to_cls = {}
                for fid,datas in fid_datas.items():
                    tmp_data = []
                    for _, tid, x, y, w, h, mark, label, _ in datas:
                        fid = int(fid)
                        tid = int(tid)
                        if not tid in tid_to_cls:
                            self.tid_curr += 1
                            tid_to_cls[tid] = self.tid_curr
                            cls = self.tid_curr
                        else:
                            cls = tid_to_cls[tid]
                        xmin = x/seq_width
                        ymin = y/seq_height
                        xmax = (x+w)/seq_width
                        ymax = (y+h)/seq_height
                        tmp_data.append([cls,[ymin,xmin,ymax,xmax]])

                    if len(tmp_data)>0:
                        img_name = '{:06d}.jpg'.format(fid)
                        img_path = os.path.join(seq_root,seq,"img1",img_name)
                        img_data = {'img_width':seq_width,'img_height':seq_height,'img_path':img_path}
                        yield img_data,tmp_data

        print(f"Last tid value {self.tid_curr}")

def create_tf_example(image,
                      annotations):
    global src_file_index
    image_height = image['img_height']
    image_width = image['img_width']
    img_path = image['img_path']

    if RECORD_IMG_SIZE is None:
        with tf.gfile.GFile(img_path, 'rb') as fid:
            encoded_jpg = fid.read()
    else:
        img = wmli.imread(img_path)
        img = wmli.resize_img(img,RECORD_IMG_SIZE,keep_aspect_ratio=True)
        encoded_jpg = wmli.encode_img(img)

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_ids = []

    for l,box in annotations:
        xmin.append(box[1])
        xmax.append(box[3])
        ymin.append(box[0])
        ymax.append(box[2])
        is_crowd.append(False)
        category_ids.append(l)

    if len(xmin)==0:
        return None

    feature_dict = {
        'image/height':
            dataset_util.int64_feature(image_height),
        'image/width':
            dataset_util.int64_feature(image_width),
        'image/filename':
            dataset_util.bytes_feature(img_path.encode('utf8')),
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/format':
            dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/label':
            dataset_util.int64_list_feature(category_ids),
        'image/object/is_crowd':
            dataset_util.int64_list_feature(is_crowd),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def _create_tf_record(
        data_dirs, output_path,
        name="mot_train"):
    if tf.gfile.IsDirectory(output_path):
        print(f"Remove {output_path}")
        shutil.rmtree(output_path)
    if not tf.gfile.IsDirectory(output_path):
        tf.gfile.MakeDirs(output_path)
    datasets = MOTDatasets(data_dirs)
    writer = None
    fidx = 0
    file_img_nr = 0
    for idx,data in enumerate(datasets.get_data_items()):
        img_data,bboxes_datas = data
        if writer is None or file_img_nr>IMAGE_PER_RECORD:
            fidx += 1
            file_img_nr = 0
            save_path = get_save_path(output_path, name, fidx)
            if writer is not None:
                writer.close()
            writer = tf.python_io.TFRecordWriter(save_path)

        if idx % 100 == 0:
            tf.logging.info('On image %d', idx)
        tf_example = create_tf_example(
            img_data,bboxes_datas)
        if tf_example is not None:
            writer.write(tf_example.SerializeToString())
            file_img_nr += 1

    if writer is not None:
        writer.close()

    tf.logging.info('Finished writing')


def get_save_path(save_dir, name, fidx):
    return os.path.join(save_dir, f"{name}_{fidx}.tfrecord")


def main(_):
    data_dirs = [
        "/home/wj/ai/mldata/MOT/MOT17/train",
        "/home/wj/ai/mldata/MOT/MOT17/test",
        "/home/wj/ai/mldata/MOT/MOT15/train",
        "/home/wj/ai/mldata/MOT/MOT15/test",
        "/home/wj/ai/mldata/MOT/MOT20/train",
    ]

    train_output_path = FLAGS.output_dir
    if not tf.gfile.IsDirectory(train_output_path):
        tf.gfile.MakeDirs(train_output_path)

    _create_tf_record(
        data_dirs,
        train_output_path,
        name='mot_train')

if __name__ == '__main__':
    FLAGS.output_dir = wmlu.home_dir("ai/mldata/MOT/tfdata_mot_train")
    tf.app.run()
