#coding=utf-8
import tensorflow as tf
import sys
import os
import numpy as np
from collections import OrderedDict
from semantic.visualization_utils import STANDARD_COLORS
import semantic.toolkit as smt
import wml_tfutils as wmlt
import time
from iotoolkit.coco_toolkit import *
import wml_utils as wmlu
import logging
import glob
from object_detection2.standard_names import *
from collections import Iterable
from object_detection2.data.transforms.transform import WTransformList
from object_detection2.standard_names import *

slim = tf.contrib.slim
slim_example_decoder = tf.contrib.slim.tfexample_decoder


def __decode_png_instance_masks(keys_to_tensors):
    def decode_png_mask(image_buffer):
        image = tf.squeeze(
            tf.image.decode_image(image_buffer, channels=1), axis=2)
        image.set_shape([None, None])
        image = tf.to_float(tf.greater(image, 0))
        return image

    png_masks = keys_to_tensors['image/object/mask']
    height = keys_to_tensors['image/height']
    width = keys_to_tensors['image/width']
    if isinstance(png_masks, tf.SparseTensor):
        png_masks = tf.sparse_tensor_to_dense(png_masks, default_value='')
    return tf.cond(
        tf.greater(tf.size(png_masks), 0),
        lambda: tf.map_fn(decode_png_mask, png_masks, dtype=tf.float32),
        lambda: tf.zeros(tf.to_int32(tf.stack([0, height, width]))))

def __parse_func(example_proto):
    keys_to_features = {
        'image/encoded':tf.FixedLenFeature((), tf.string, default_value=''), #[H,W,C]
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature((), tf.int64,1),
        'image/width': tf.FixedLenFeature((), tf.int64,1),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/is_crowd':tf.VarLenFeature(dtype=tf.int64),
        'image/object/mask': tf.VarLenFeature(tf.string), #[N,H,W], value=0,1
        'image/file_index': tf.FixedLenFeature((), tf.int64,1)
    }
    example_proto = tf.parse_single_example(example_proto,keys_to_features)
    image = tf.image.decode_jpeg(example_proto['image/encoded'],channels=3)
    masks = __decode_png_instance_masks(example_proto)
    xmin = tf.sparse_tensor_to_dense(example_proto['image/object/bbox/xmin'])
    ymin = tf.sparse_tensor_to_dense(example_proto['image/object/bbox/ymin'])
    xmax = tf.sparse_tensor_to_dense(example_proto['image/object/bbox/xmax'])
    ymax = tf.sparse_tensor_to_dense(example_proto['image/object/bbox/ymax'])
    xmin = tf.reshape(xmin,[-1,1])
    ymin = tf.reshape(ymin,[-1,1])
    xmax = tf.reshape(xmax,[-1,1])
    ymax = tf.reshape(ymax,[-1,1])
    boxes = tf.concat([ymin,xmin,ymax,xmax],axis=1)
    keys_to_res = {
        IMAGE:image,
        GT_MASKS:masks,
        HEIGHT:'image/height',
        WIDTH:'image/width',
        ORG_HEIGHT:'image/height',
        ORG_WIDTH:'image/width',
        GT_BOXES:boxes,
        GT_LABELS:'image/object/class/label',
        IS_CROWD:'image/object/is_crowd',
        FILEINDEX:'image/file_index',
    }
    res = {}
    for k,v in keys_to_res.items():
        if isinstance(v,str):
            if isinstance(example_proto[v],tf.SparseTensor):
                res[k] = tf.sparse_tensor_to_dense(example_proto[v])
            else:
                res[k] = example_proto[v]
        else:
            res[k] = v
    return res

'''
file_pattern类似于 'voc_2012_%s_*.tfrecord'
split_to_sizes:所有文件中样本的数量
items_to_descriptions:解码的数据项的描述
num_classes:类别数，不包含背景
'''
def get_database(dataset_dir,num_parallel=1,file_pattern='*_train.record'):

    file_pattern = os.path.join(dataset_dir,file_pattern)
    files = glob.glob(file_pattern)
    if len(files) == 0:
        logging.error(f'No files found in {file_pattern}')
        a = input(f"No files found in {file_pattern}, continue?(y/n)")
        if a != 'y':
            exit(-1)
    else:
        print(f"Total {len(files)} files.")
        wmlu.show_list(files)
    dataset = tf.data.TFRecordDataset(files,num_parallel_reads=num_parallel)
    dataset = dataset.map(__parse_func,num_parallel_calls=num_parallel)
    return dataset
def get_from_dataset(data_item,keys):
    res = []
    for k in keys:
        res.append(data_item[k])
    return res
'''
return:
image:[H,W,C]
labels:[X]
bboxes:[X,4]
mask:[X,H,W]
'''
def get_data(data_dir,num_parallel=8,log_summary=True,file_pattern="*.tfrecord",id_to_label={},transforms=None,has_file_index=True,filter_empty=True):
    '''
    id_to_label:first id is the category_id in coco, second label is the label id for model
    '''
    dataset = get_database(dataset_dir=data_dir,num_parallel=num_parallel,file_pattern=file_pattern)
    def filter_func(x):
        return tf.greater(tf.shape(x[GT_BOXES])[0],0)
    if filter_empty:
        dataset = dataset.filter(filter_func)
    if transforms is not None:
        if isinstance(transforms,Iterable):
            transforms = WTransformList(transforms)
        dataset = dataset.map(transforms,num_parallel_calls=num_parallel)
    if filter_empty:
        #处理后有的bbox可能消失了
        dataset = dataset.filter(filter_func)
    with tf.name_scope('data_provider'):
        pass
    return dataset
