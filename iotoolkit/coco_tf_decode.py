#coding=utf-8
import tensorflow as tf
import sys
import os
import numpy as np
from collections import OrderedDict
import object_detection.utils as odu
from semantic.visualization_utils import STANDARD_COLORS
import semantic.toolkit as smt
import wml_tfutils as wmlt
import time
from iotoolkit.coco_toolkit import *

slim = tf.contrib.slim
slim_example_decoder = tf.contrib.slim.tfexample_decoder


def _decode_png_instance_masks(keys_to_tensors):
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
def _reshape_instance_masks(keys_to_tensors):
    """Reshape instance segmentation masks.

    The instance segmentation masks are reshaped to [num_instances, height,
    width].

    Args:
      keys_to_tensors: a dictionary from keys to tensors.

    Returns:
      A 3-D float tensor of shape [num_instances, height, width] with values
        in {0, 1}.
    """
    height = keys_to_tensors['image/height']
    width = keys_to_tensors['image/width']
    to_shape = tf.cast(tf.stack([-1, height, width]), tf.int32)
    masks = keys_to_tensors['image/object/mask']
    if isinstance(masks, tf.SparseTensor):
      masks = tf.sparse_tensor_to_dense(masks)
    masks = tf.reshape(tf.to_float(tf.greater(masks, 0.0)), to_shape)
    return tf.cast(masks, tf.float32)

'''
file_pattern类似于 'voc_2012_%s_*.tfrecord'
split_to_sizes:所有文件中样本的数量
items_to_descriptions:解码的数据项的描述
num_classes:类别数，不包含背景
'''
def get_database(dataset_dir,num_samples=1,file_pattern='*_train.record',
              items_to_descriptions=None, num_classes=4):

    file_pattern = os.path.join(dataset_dir,file_pattern)
    reader = tf.TFRecordReader
    keys_to_features = {
        'image/encoded':tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature((), tf.int64,1),
        'image/width': tf.FixedLenFeature((), tf.int64,1),
        'image/file_index': tf.FixedLenFeature((), tf.int64,1),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/mask': tf.VarLenFeature(tf.string)
        }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format',channels=3),
        'height': slim.tfexample_decoder.Tensor('image/height'),
        'width': slim.tfexample_decoder.Tensor('image/width'),
        'file_index': slim.tfexample_decoder.Tensor('image/file_index'),
        'mask': slim_example_decoder.ItemHandlerCallback(
            ['image/object/mask', 'image/height', 'image/width'],
            _decode_png_instance_masks),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
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
'''
return:
image:[H,W,C]
labels:[X]
bboxes:[X,4]
mask:[X,H,W]
'''
def get_data(data_dir,batch_size,num_samples=1,num_classes=80,log_summary=True,file_pattern="*.tfrecord",id_to_label={}):
    '''
    id_to_label:first id is the category_id in coco, second label is the label id for model
    '''
    dataset = get_database(dataset_dir=data_dir,num_classes=num_classes,num_samples=num_samples,file_pattern=file_pattern)
    with tf.name_scope('data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=4,
            common_queue_capacity=10 * batch_size,
            common_queue_min=3 * batch_size,
            seed=int(time.time()),
            shuffle=True)
        [image, labels, bboxes,height,width,mask,file_index] = provider.get(["image", "object/label", "object/bbox","height","width","mask","file_index"])
        m_shape = tf.shape(image)
        labels,mask = tf.cond(tf.greater(tf.shape(labels)[0],0),lambda :(labels,mask),lambda :
        (tf.constant([0],dtype=tf.int64),tf.ones([1,m_shape[0],m_shape[1]],dtype=tf.float32)))
        size = tf.stack([height,width,3],axis=0)
        image = tf.reshape(image,shape=size)
        mask = tf.reshape(mask,shape=tf.stack([-1,height,width],axis=0))
        if log_summary:
            odu.tf_summary_image_with_box(image,bboxes)
            wmlt.variable_summaries_v2(mask,"mask")
            wmlt.variable_summaries_v2(image,"image")
            wmlt.variable_summaries_v2(file_index,"file_index")

        dict = OrderedDict(ID_TO_TEXT)
        id_to_color = {} #id is the category_id, color is string
        label = 1
        if len(id_to_label)==0:
            for key in dict.keys():
                id_to_label[key] = label
                id_to_color[key] = STANDARD_COLORS[label-1]
                label += 1

        for key in dict.keys():
            if key in id_to_label:
                label = id_to_label[key]
            else:
                label = len(STANDARD_COLORS)
            id_to_color[key] = STANDARD_COLORS[label-1]
        table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(np.array(list(id_to_label.keys()), dtype=np.int64),
                                                        np.array(list(id_to_label.values()), dtype=np.int64)), -1)
        color_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(np.array(list(id_to_color.keys()), dtype=np.int64),
                                                        np.array(list(id_to_color.values()), dtype=np.str)), "Red")
        labels = table.lookup(labels)
        if log_summary:
            wmlt.variable_summaries_v2(labels,"labels")
            colors = color_table.lookup(labels)
            smt.tf_summary_image_with_mask(image, mask, colors)

        wmlt.add_to_hash_table_collection(table.init)
        wmlt.add_to_hash_table_collection(color_table.init)

    return image,labels,bboxes,mask
