#coding=utf-8
import os
import sys
import random
import time
import numpy as np
import tensorflow as tf
import object_detection.utils as odu
import object_detection.npod_toolkit as npod

import shutil
import xml.etree.ElementTree as ET
import sys
sys.path.append("/Users/vghost/MachineLearning/model/wml")
from wml_tfutils import int64_feature,bytes_feature

tf.flags.DEFINE_string('data_dir', '',
                       'Training image directory.')
tf.flags.DEFINE_string('val_image_dir', '',
                       'Validation image directory.')
tf.flags.DEFINE_string('test_image_dir', '',
                       'Test image directory.')
tf.flags.DEFINE_string('train_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_annotations_file', '',
                       'Validation annotations JSON file.')
tf.flags.DEFINE_string('testdev_annotations_file', '',
                       'Test-dev annotations JSON file.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'
SAMPLES_PER_FILES = 200

def category_id_filter(category_id):
    good_ids = [15,6,7,14,2]
    return category_id in good_ids

def labels_text_to_labels(labels_text):
    return [1]*len(labels_text)

def is_good_data(labels):
    for label in labels:
        if category_id_filter(label):
            return True

    return False

'''
directory:图像目录路径
name:图像文件名，但不包含路径及文件名

返回图像数据，bbox(用[0,1]表示，bbox相对应的label
'''
def _process_image(directory, name):
    filename = os.path.join(directory,DIRECTORY_IMAGES + name + '.jpg')
    image_data = tf.gfile.FastGFile(filename, 'r').read()

    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    shape, bboxes, labels_text, difficult, truncated = odu.read_voc_xml(filename, adjust=None)
    labels = labels_text_to_labels(labels_text)
    if not is_good_data(labels):
        return None,None,None,None,None,None,None
    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


def _convert_to_example(image_data, labels, labels_text, bboxes, shape,
                        difficult, truncated):
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        #每一个bbox应当包含四个元素(ymin,xmin,ymax,xmax)
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        #l依次为ymin,xmin,ymax,xmax,point依次为b.ymin,b.xmin,b.ymax,b.xmax
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned
    '''
    现在xmin,ymin,xmax,ymax包含了所有在bboxes中的数据
    '''
    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example

'''
dataset_dir:图像目录路径
name:图像文件名，不包含路径及后辍
'''
def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(dataset_dir, name)
    if image_data is None:
        return False
    example = _convert_to_example(image_data, labels, labels_text,
                                  bboxes, shape, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())
    return True


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)

'''
将所有图像文件按SAMPLES_PER_FILES(200)一批保存在tfrecored文件中
'''
def to_tfrecords(dataset_dir, output_dir, name='train', shuffling=False):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print("删除文件夹%s"%(output_dir))
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    '''
    filenames仅包含文件名，不包含文件路径
    '''
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(time.time())
        random.shuffle(filenames)

    i = 0
    fidx = 0
    while i < len(filenames):
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img_name = filename[:-4]
                if _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer):
                    j += 1
                i += 1

            fidx += 1
    print('\nFinished converting the dataset total %d examples.!'%(len(filenames)))

if __name__ == "__main__":

    dataset_dir = "/Users/vghost/MachineLearning/mldata/dentalfilm/fullviewod_jpgdatav8"
    output_dir = "/Users/vghost/MachineLearning/mldata/dentalfilm/fullviewod_tfdata"
    output_name = "train"

    print('Dataset directory:', dataset_dir)
    print('Output directory:',output_dir)

    to_tfrecords(dataset_dir, output_dir, output_name)