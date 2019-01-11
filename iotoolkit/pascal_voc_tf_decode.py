#coding=utf-8
import os
import tensorflow as tf
import trainprocess as trainpre
import os
import numpy as np
import wml_tfutils as wmlt
import object_detection.utils as odu

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim
CROP_IMAGE_WIDTH=2800
CROP_IMAGE_HEIGHT=1500
IMAGE_WIDTH=720
IMAGE_HEIGHT=384
MAX_BOXES_NR=36 #boxes number in one example

slim = tf.contrib.slim


'''
file_pattern类似于 'voc_2012_%s_*.tfrecord'
split_to_sizes:所有文件中样本的数量
items_to_descriptions:解码的数据项的描述
num_classes:类别数，不包含背景
'''

def get_database(dataset_dir,num_samples=1,file_pattern='train_*.tfrecord',
              items_to_descriptions=None, num_classes=4):

    file_pattern = os.path.join(dataset_dir,file_pattern)
    reader = tf.TFRecordReader
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format',channels=1),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=num_samples,
            items_to_descriptions=items_to_descriptions,
            num_classes=num_classes,
            labels_to_names=None)

def get_data(data_dir,num_samples=1,num_classes=3,id_to_label=[]):
    dataset = get_database(dataset_dir=data_dir,num_classes=num_classes,num_samples=num_samples)
    with tf.name_scope('data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=2,
            common_queue_capacity=20 * FLAGS.batch_size,
            common_queue_min=10 * FLAGS.batch_size,
            shuffle=True)
        [image, glabels, bboxes] = provider.get(["image", "object/label", "object/bbox"])
    label = 1
    if len(id_to_label) == 0:
        for key in dict.keys():
            id_to_label[key] = label
            label += 1
    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(np.array(list(id_to_label.keys()), dtype=np.int64),
                                                    np.array(list(id_to_label.values()), dtype=np.int64)), -1)
    glabels = table.lookup(glabels)
    wmlt.add_to_hash_table_collection(table.init)

    return image,glabels,bboxes

