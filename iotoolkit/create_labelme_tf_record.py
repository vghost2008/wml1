#coding=utf-8
import json
import wml_utils as wmlu
import os
import cv2 as cv
import numpy as np
import shutil
import hashlib
import io
import json
import os
import numpy as np
import PIL.Image
import wml_utils as wmlu
import object_detection.bboxes as odb
import random
import tensorflow as tf
import time
import iotoolkit.dataset_util as dataset_util
import sys
import img_utils as wmli
import copy
import iotoolkit.label_map_util as label_map_util
from iotoolkit.labelme_toolkit import *
from multiprocessing import Pool
import functools
import cv2

flags = tf.app.flags
tf.flags.DEFINE_string('data_dir', '',
                       'data directory.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

TRAIN_SIZE_LIMIT = None
VAL_SIZE_LIMIT = None
SAMPLES_PER_FILES = 200

def category_id_filter(category_id):
    good_ids = [1,2,3]
    return category_id in good_ids

text_to_id={"a":1,"b":2,"c":3,"d":4}
USE_INDEX_IN_FILE = False
#text_to_id={"0":1,"1":2,"2":3}
def label_text_to_id(text):
    return text_to_id[text]

def create_tf_example(image,
                      annotations_list,
                      img_file,
                      id):
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    if USE_INDEX_IN_FILE:
        file_index = int(filename[filename.find("_")+1:])
    else:
        file_index = id

    with tf.gfile.GFile(img_file, 'rb') as fid:
        encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    category_ids = []
    encoded_mask_png = []
    num_annotations_skipped = 0
    for object_annotations in annotations_list:
        (x, y, width, height) = tuple(object_annotations['bbox'])
        if width <= 0 or height <= 0:
          num_annotations_skipped += 1
          continue
        if x + width > image_width or y + height > image_height:
          num_annotations_skipped += 1
          continue
        category_id = int(object_annotations['category_id'])
        if not category_id_filter(category_id):
          num_annotations_skipped += 1
          continue

        xmin.append(float(x) / image_width)
        xmax.append(float(x + width) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(y + height) / image_height)
        category_ids.append(category_id)

        binary_mask = object_annotations["segmentation"]
        #cv2.imwrite(wmlu.home_dir("x.jpg"),binary_mask*255)
        pil_image = PIL.Image.fromarray(binary_mask)
        output_io = io.BytesIO()
        pil_image.save(output_io, format='PNG')
        encoded_mask_png.append(output_io.getvalue())

    #for test
    #if len(xmin) == 0:
        #return None,None

    feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
       'image/file_index':dataset_util.int64_feature(file_index),
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
    }
    feature_dict['image/object/mask'] = (
        dataset_util.bytes_list_feature(encoded_mask_png))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    if example is None:
        print("None example")
        return None,None
    return example, num_annotations_skipped

def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)

def _add_to_tfrecord(file,writer,id):
    img_file,json_file = file
    image_info,annotations_list = read_labelme_data(json_file,label_text_to_id)
    if len(annotations_list)>32:
        print(f"Too many annotations_list {len(annotations_list)}")
        return False
    if len(annotations_list)==0:
        print("No annotations.")
        #for test
        #raise RuntimeError("No annotations.")
        #return False
    tf_example, num_annotations_skipped = create_tf_example(
        image_info, annotations_list,img_file,id)
    if tf_example is not None:
        writer.write(tf_example.SerializeToString())
        writer.flush()
        return True
    return False


def _create_tf_record(data_dir,output_dir,img_suffix="jpg",name="train",shuffling=True,fidx=0):
    files = get_files(data_dir,img_suffix=img_suffix)
    if os.path.exists(output_dir) and (data_dir != output_dir):
        shutil.rmtree(output_dir)
        print("删除文件夹%s"%(output_dir))
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    if shuffling:
        random.seed(time.time())
        random.shuffle(files)

    i = 0
    while i < len(files):
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(files) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(files)))
                sys.stdout.flush()

                if _add_to_tfrecord(files[i], tfrecord_writer):
                    j += 1
                i += 1

            fidx += 1
    print('\nFinished converting the dataset total %d examples.!'%(len(files)))

def make_tfrecord(file_data,output_dir,name):
    fidx,file_d = file_data
    tf_filename = _get_output_filename(output_dir, name, fidx)
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i,file in enumerate(file_d):
            _add_to_tfrecord(file, tfrecord_writer,id=fidx*SAMPLES_PER_FILES+i)

def multithread_create_tf_record(data_dir,output_dir,img_suffix="jpg",name="train",shuffling=True,fidx=0):
    files = get_files(data_dir,img_suffix=img_suffix)
    if os.path.exists(output_dir) and (data_dir != output_dir):
        shutil.rmtree(output_dir)
        print("删除文件夹%s"%(output_dir))
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    if shuffling:
        random.seed(time.time())
        random.shuffle(files)
    files = wmlu.list_to_2dlist(files,SAMPLES_PER_FILES)
    files_data = list(enumerate(files))
    if fidx != 0:
        _files_data = []
        for fid,file_d in files_data:
            _files_data.append([fid+fidx,file_d])
        files_data = _files_data

    pool = Pool(10)
    pool.map(functools.partial(make_tfrecord,output_dir=output_dir,name=name),files_data)
    pool.close()
    pool.join()

    print('\nFinished converting the dataset total %d examples.!'%(len(files)))


if __name__ == "__main__":

    #dataset_dir = "/home/vghost/ai/mldata/mnistod/eval"
    #output_dir = "/home/vghost/ai/mldata/mnistod/eval_tfrecord"
    dataset_dir = "/home/vghost/ai/mldata/mnistod/train1"
    output_dir = "/home/vghost/ai/mldata/mnistod/train1_tfrecord"
    output_name = "train"

    print('Dataset directory:', dataset_dir)
    print('Output directory:',output_dir)
    random.seed(int(time.time()))

    multithread_create_tf_record(dataset_dir, output_dir,fidx=0)
