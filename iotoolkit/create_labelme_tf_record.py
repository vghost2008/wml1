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
import random
import tensorflow as tf
import time
import iotoolkit.dataset_util as dataset_util
import sys
import iotoolkit.label_map_util as label_map_util

flags = tf.app.flags
tf.flags.DEFINE_string('data_dir', '',
                       'data directory.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

TRAIN_SIZE_LIMIT = None
VAL_SIZE_LIMIT = None
SAMPLES_PER_FILES = 100

def category_id_filter(category_id):
    good_ids = [1,2,3,4,6,8]
    return category_id in good_ids

text_to_id={"a":1,"b":2,"c":3}
def label_text_to_id(text):
    return text_to_id[text]


def get_files(data_dir,img_suffix="jpg"):
    files = wmlu.recurse_get_filepath_in_dir(data_dir,suffix=".json")
    res = []
    for file in files:
        img_file = wmlu.change_suffix(file,img_suffix)
        if os.path.exists(img_file):
            res.append((img_file,file))
    
    return res

def read_labelme_data(file_path):
    annotations_list = []
    image = {}
    with open(file_path) as f:
        json_data = json.load(f)
        img_width = json_data["imageWidth"]
        img_height = json_data["imageHeight"]
        image["height"] = img_height
        image["width"] = img_width
        image["file_name"] = wmlu.base_name(file_path)
        for shape in json_data["shapes"]:
            mask = np.zeros(shape=[img_height,img_width],dtype=np.uint8)
            all_points = np.array([shape["points"]])
            points = np.transpose(all_points[0])
            x,y = np.vsplit(points,2)
            xmin = np.min(x)
            xmax = np.max(x)
            ymin = np.min(y)
            ymax = np.max(y)
            segmentation = cv.drawContours(mask,all_points,-1,color=(1))
            label = label_text_to_id(shape["label"])
            annotations_list.append({"bbox":(xmin,ymin,xmax-xmin+1,ymax-ymin+1),"segmentation":segmentation,"category_id":label})
    return image,annotations_list
        



def create_tf_example(image,
                      annotations_list,
                      img_file):
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']

    with tf.gfile.GFile(img_file, 'rb') as fid:
        encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
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
        pil_image = PIL.Image.fromarray(binary_mask)
        output_io = io.BytesIO()
        pil_image.save(output_io, format='PNG')
        encoded_mask_png.append(output_io.getvalue())
        feature_dict = {
          'image/height':
              dataset_util.int64_feature(image_height),
          'image/width':
              dataset_util.int64_feature(image_width),
          'image/filename':
              dataset_util.bytes_feature(filename.encode('utf8')),
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
        #category_id is the key of ID_TO_TEXT
        if len(category_ids)==0:
          return None,None,None
    return example, num_annotations_skipped

def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)

def _add_to_tfrecord(file,writer):
    img_file,json_file = file
    image_info,annotations_list = read_labelme_data(json_file)
    tf_example, num_annotations_skipped = create_tf_example(
        image_info, annotations_list,img_file)
    if tf_example is not None:
        writer.write(tf_example.SerializeToString())
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



if __name__ == "__main__":

    dataset_dir = "/home/vghost/workfile/test"
    output_dir = "/home/vghost/workfile/test/output"
    output_name = "train"

    print('Dataset directory:', dataset_dir)
    print('Output directory:',output_dir)

    _create_tf_record(dataset_dir, output_dir)
