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

slim = tf.contrib.slim
slim_example_decoder = tf.contrib.slim.tfexample_decoder
ID_TO_TEXT={1: {u'supercategory': u'person', u'id': 1, u'name': u'person'},
            2: {u'supercategory': u'vehicle', u'id': 2, u'name': u'bicycle'},
            3: {u'supercategory': u'vehicle', u'id': 3, u'name': u'car'},
            4: {u'supercategory': u'vehicle', u'id': 4, u'name': u'motorcycle'},
            5: {u'supercategory': u'vehicle', u'id': 5, u'name': u'airplane'},
            6: {u'supercategory': u'vehicle', u'id': 6, u'name': u'bus'},
            7: {u'supercategory': u'vehicle', u'id': 7, u'name': u'train'},
            8: {u'supercategory': u'vehicle', u'id': 8, u'name': u'truck'},
            9: {u'supercategory': u'vehicle', u'id': 9, u'name': u'boat'},
            10: {u'supercategory': u'outdoor', u'id': 10, u'name': u'traffic light'},
            11: {u'supercategory': u'outdoor', u'id': 11, u'name': u'fire hydrant'},
            13: {u'supercategory': u'outdoor', u'id': 13, u'name': u'stop sign'},
            14: {u'supercategory': u'outdoor', u'id': 14, u'name': u'parking meter'},
            15: {u'supercategory': u'outdoor', u'id': 15, u'name': u'bench'},
            16: {u'supercategory': u'animal', u'id': 16, u'name': u'bird'},
            17: {u'supercategory': u'animal', u'id': 17, u'name': u'cat'},
            18: {u'supercategory': u'animal', u'id': 18, u'name': u'dog'},
            19: {u'supercategory': u'animal', u'id': 19, u'name': u'horse'},
            20: {u'supercategory': u'animal', u'id': 20, u'name': u'sheep'},
            21: {u'supercategory': u'animal', u'id': 21, u'name': u'cow'},
            22: {u'supercategory': u'animal', u'id': 22, u'name': u'elephant'},
            23: {u'supercategory': u'animal', u'id': 23, u'name': u'bear'},
            24: {u'supercategory': u'animal', u'id': 24, u'name': u'zebra'},
            25: {u'supercategory': u'animal', u'id': 25, u'name': u'giraffe'},
            27: {u'supercategory': u'accessory', u'id': 27, u'name': u'backpack'},
            28: {u'supercategory': u'accessory', u'id': 28, u'name': u'umbrella'},
            31: {u'supercategory': u'accessory', u'id': 31, u'name': u'handbag'},
            32: {u'supercategory': u'accessory', u'id': 32, u'name': u'tie'},
            33: {u'supercategory': u'accessory', u'id': 33, u'name': u'suitcase'},
            34: {u'supercategory': u'sports', u'id': 34, u'name': u'frisbee'},
            35: {u'supercategory': u'sports', u'id': 35, u'name': u'skis'},
            36: {u'supercategory': u'sports', u'id': 36, u'name': u'snowboard'},
            37: {u'supercategory': u'sports', u'id': 37, u'name': u'sports ball'},
            38: {u'supercategory': u'sports', u'id': 38, u'name': u'kite'},
            39: {u'supercategory': u'sports', u'id': 39, u'name': u'baseball bat'},
            40: {u'supercategory': u'sports', u'id': 40, u'name': u'baseball glove'},
            41: {u'supercategory': u'sports', u'id': 41, u'name': u'skateboard'},
            42: {u'supercategory': u'sports', u'id': 42, u'name': u'surfboard'},
            43: {u'supercategory': u'sports', u'id': 43, u'name': u'tennis racket'},
            44: {u'supercategory': u'kitchen', u'id': 44, u'name': u'bottle'},
            46: {u'supercategory': u'kitchen', u'id': 46, u'name': u'wine glass'},
            47: {u'supercategory': u'kitchen', u'id': 47, u'name': u'cup'},
            48: {u'supercategory': u'kitchen', u'id': 48, u'name': u'fork'},
            49: {u'supercategory': u'kitchen', u'id': 49, u'name': u'knife'},
            50: {u'supercategory': u'kitchen', u'id': 50, u'name': u'spoon'},
            51: {u'supercategory': u'kitchen', u'id': 51, u'name': u'bowl'},
            52: {u'supercategory': u'food', u'id': 52, u'name': u'banana'},
            53: {u'supercategory': u'food', u'id': 53, u'name': u'apple'},
            54: {u'supercategory': u'food', u'id': 54, u'name': u'sandwich'},
            55: {u'supercategory': u'food', u'id': 55, u'name': u'orange'},
            56: {u'supercategory': u'food', u'id': 56, u'name': u'broccoli'},
            57: {u'supercategory': u'food', u'id': 57, u'name': u'carrot'},
            58: {u'supercategory': u'food', u'id': 58, u'name': u'hot dog'},
            59: {u'supercategory': u'food', u'id': 59, u'name': u'pizza'},
            60: {u'supercategory': u'food', u'id': 60, u'name': u'donut'},
            61: {u'supercategory': u'food', u'id': 61, u'name': u'cake'},
            62: {u'supercategory': u'furniture', u'id': 62, u'name': u'chair'},
            63: {u'supercategory': u'furniture', u'id': 63, u'name': u'couch'},
            64: {u'supercategory': u'furniture', u'id': 64, u'name': u'potted plant'},
            65: {u'supercategory': u'furniture', u'id': 65, u'name': u'bed'},
            67: {u'supercategory': u'furniture', u'id': 67, u'name': u'dining table'},
            70: {u'supercategory': u'furniture', u'id': 70, u'name': u'toilet'},
            72: {u'supercategory': u'electronic', u'id': 72, u'name': u'tv'},
            73: {u'supercategory': u'electronic', u'id': 73, u'name': u'laptop'},
            74: {u'supercategory': u'electronic', u'id': 74, u'name': u'mouse'},
            75: {u'supercategory': u'electronic', u'id': 75, u'name': u'remote'},
            76: {u'supercategory': u'electronic', u'id': 76, u'name': u'keyboard'},
            77: {u'supercategory': u'electronic', u'id': 77, u'name': u'cell phone'},
            78: {u'supercategory': u'appliance', u'id': 78, u'name': u'microwave'},
            79: {u'supercategory': u'appliance', u'id': 79, u'name': u'oven'},
            80: {u'supercategory': u'appliance', u'id': 80, u'name': u'toaster'},
            81: {u'supercategory': u'appliance', u'id': 81, u'name': u'sink'},
            82: {u'supercategory': u'appliance', u'id': 82, u'name': u'refrigerator'},
            84: {u'supercategory': u'indoor', u'id': 84, u'name': u'book'},
            85: {u'supercategory': u'indoor', u'id': 85, u'name': u'clock'},
            86: {u'supercategory': u'indoor', u'id': 86, u'name': u'vase'},
            87: {u'supercategory': u'indoor', u'id': 87, u'name': u'scissors'},
            88: {u'supercategory': u'indoor', u'id': 88, u'name': u'teddy bear'},
            89: {u'supercategory': u'indoor', u'id': 89, u'name': u'hair drier'},
            90: {u'supercategory': u'indoor', u'id': 90, u'name': u'toothbrush'}}

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
            num_readers=2,
            common_queue_capacity=20 * batch_size,
            common_queue_min=3 * batch_size,
            seed=int(time.time()),
            shuffle=True)
        [image, labels, bboxes,height,width,mask] = provider.get(["image", "object/label", "object/bbox","height","width","mask"])
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
