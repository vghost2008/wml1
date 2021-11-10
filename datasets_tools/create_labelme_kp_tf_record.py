# coding=utf-8
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
SAMPLES_PER_FILES = 500


def category_id_filter(category_id):
    '''good_ids = [1,2,3]
    res = category_id in good_ids'''
    res = True
    if not res:
        print(f"bad category id {category_id}")
    return res


USE_INDEX_IN_FILE = False


def create_tf_example(image,
                      labels,
                      points,
                      img_file,
                      id):
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    if USE_INDEX_IN_FILE:
        file_index = int(filename[filename.find("_") + 1:])
    else:
        file_index = id

    with tf.gfile.GFile(img_file, 'rb') as fid:
        encoded_jpg = fid.read()

    xs = []
    ys = []
    category_ids = []
    encoded_mask_png = []
    num_annotations_skipped = 0
    print("ann size:", len(labels))
    for label,point in zip(labels,points):
        x,y = point
        category_id = int(label)
        if not category_id_filter(category_id):
            num_annotations_skipped += 1
            continue

        xs.append(float(x) / image_width)
        ys.append(float(y) / image_height)
        category_ids.append(category_id)

    feature_dict = {
        'image/height':
            dataset_util.int64_feature(image_height),
        'image/width':
            dataset_util.int64_feature(image_width),
        'image/filename':
            dataset_util.bytes_feature(filename.encode('utf8')),
        'image/file_index': dataset_util.int64_feature(file_index),
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/format':
            dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/point/x':
            dataset_util.float_list_feature(xs),
        'image/object/point/y':
            dataset_util.float_list_feature(ys),
        'image/object/class/label':
            dataset_util.int64_list_feature(category_ids),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    if example is None:
        print("None example")
        return None, None
    return example, num_annotations_skipped


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def _add_to_tfrecord(file, writer, id, label_text_to_id):
    img_file, json_file = file
    image_info, labels,points = read_labelme_kp_data(json_file, label_text_to_id)
    if len(labels) == 0:
        print("No annotations.")
        # for test
        # raise RuntimeError("No annotations.")
        # return False
    tf_example, num_annotations_skipped = create_tf_example(
        image_info, labels,points, img_file, id)
    if tf_example is not None:
        writer.write(tf_example.SerializeToString())
        writer.flush()
        return True
    return False


def _create_tf_record(data_dir, output_dir, img_suffix="jpg", name="train", shuffling=True, fidx=0,
                      label_text_to_id=None):
    files = get_files(data_dir, img_suffix=img_suffix)
    if os.path.exists(output_dir) and (data_dir != output_dir):
        shutil.rmtree(output_dir)
        print("删除文件夹%s" % (output_dir))
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
                sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(files)))
                sys.stdout.flush()

                if _add_to_tfrecord(files[i], tfrecord_writer, label_text_to_id=label_text_to_id):
                    j += 1
                i += 1

            fidx += 1
    print('\nFinished converting the dataset total %d examples.!' % (len(files)))


def make_tfrecord(file_data, output_dir, name, label_text_to_id):
    fidx, file_d = file_data
    tf_filename = _get_output_filename(output_dir, name, fidx)
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, file in enumerate(file_d):
            _add_to_tfrecord(file, tfrecord_writer, id=fidx * SAMPLES_PER_FILES + i, label_text_to_id=label_text_to_id)


def multithread_create_tf_record(data_dir, output_dir, img_suffix="jpg", name="train", shuffling=True, fidx=0,
                                 label_text_to_id=None):
    files = get_files(data_dir, img_suffix=img_suffix)
    if os.path.exists(output_dir) and (data_dir != output_dir):
        shutil.rmtree(output_dir)
        print("删除文件夹%s" % (output_dir))
    return multithread_create_tf_record_by_files(files, output_dir,
                                                 name, shuffling, fidx,
                                                 label_text_to_id)


def multithread_create_tf_record_by_files(files, output_dir, name="train", shuffling=True, fidx=0,
                                          label_text_to_id=None):
    print(f"samples per files {SAMPLES_PER_FILES}.")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print("删除文件夹%s" % (output_dir))
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    if shuffling:
        random.seed(time.time())
        random.shuffle(files)
    print(f"Total {len(files)} files.")
    files = wmlu.list_to_2dlist(files, SAMPLES_PER_FILES)
    files_data = list(enumerate(files))
    if fidx != 0:
        _files_data = []
        for fid, file_d in files_data:
            _files_data.append([fid + fidx, file_d])
        files_data = _files_data

    sys.stdout.flush()
    pool = Pool(13)
    pool.map(functools.partial(make_tfrecord, output_dir=output_dir, name=name, label_text_to_id=label_text_to_id),
             files_data)
    pool.close()
    pool.join()

    print('\nFinished converting the dataset total %d examples.!' % (len(files)))


def label_text_2_id(label):
    dicts = {'a':0,'b':1}
    return dicts[label]
if __name__ == "__main__":
    dataset_dir = wmlu.home_dir("ai/mldata/court_detection/example/")
    output_dir = wmlu.home_dir("ai/mldata/court_detection/train/")
    output_name = "train"

    print('Dataset directory:', dataset_dir)
    print('Output directory:', output_dir)
    random.seed(int(time.time()))

    multithread_create_tf_record(dataset_dir, output_dir, fidx=0,label_text_to_id=label_text_2_id)
