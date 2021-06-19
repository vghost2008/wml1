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

from pycocotools import mask
import tensorflow as tf

import iotoolkit.dataset_util as dataset_util
import iotoolkit.label_map_util as label_map_util
from iotoolkit.coco_toolkit import *

flags = tf.app.flags
tf.flags.DEFINE_boolean('include_masks', True,
                        'Whether to include instance segmentations masks '
                        '(PNG encoded) in the result. default: False.')
tf.flags.DEFINE_string('data_dir',wmlu.home_dir("ai/mldata/coco"),
                       'Training image directory.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

TRAIN_SIZE_LIMIT = None
VAL_SIZE_LIMIT = None
src_file_index = 0
IMAGE_PER_RECORD = 10000
COCO_KP_LEVEL = 1
COCO_KP_MIN_AREA = 1024
COCO_KP_MIN_NR = 5


def category_id_filter(category_id):
    return True


def trans_id(category_id):
    return ID_TO_COMPRESSED_ID[category_id]
    # return category_id


def reverse_trans_id(category_id):
    return COMPRESSED_ID_TO_ID[category_id]
    # return category_id

def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index,
                      include_masks=False):
    """Converts image and annotations to a tf.Example proto.

    Args:
      image: dict with keys:
        [u'license', u'file_name', u'coco_url', u'height', u'width',
        u'date_captured', u'flickr_url', u'id']
      annotations_list:
        list of dicts with keys:
        [u'segmentation', u'area', u'iscrowd', u'image_id',
        u'bbox', u'category_id', u'id']
        Notice that bounding box coordinates in the official COCO dataset are
        given as [x, y, width, height] tuples using absolute coordinates where
        x, y represent the top-left (0-indexed) corner.  This function converts
        to the format expected by the Tensorflow Object Detection API (which is
        which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
        to image size).
      image_dir: directory containing the image files.
      category_index: a dict containing COCO category information keyed
        by the 'id' field of each category.  See the
        label_map_util.create_category_index function.
      include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
    Returns:
      example: The converted tf.Example
      num_annotations_skipped: Number of (invalid) annotations that were ignored.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    global src_file_index
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    full_path = os.path.join(image_dir, filename)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    # encoded_jpg_io = io.BytesIO(encoded_jpg)
    # image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    num_annotations_skipped = 0
    keypoints_x = []
    keypoints_y = []
    repeat_nr = 1
    has_keypoints = False
    for object_annotations in annotations_list:
        (x, y, width, height) = tuple(object_annotations['bbox'])
        l_key_points = np.array(object_annotations['keypoints'])
        l_key_points = np.reshape(l_key_points,[COCO_KP_NR,3])
        num_keypoints = object_annotations['num_keypoints']

        if width <= 0 or height <= 0:
            num_annotations_skipped += 1
            continue
        if x + width > image_width or y + height > image_height:
            num_annotations_skipped += 1
            continue

        category_id = int(object_annotations['category_id'])
        category_id = trans_id(category_id)

        if not category_id_filter(category_id):
            num_annotations_skipped += 1
            continue

        if num_keypoints<COCO_KP_MIN_NR and object_annotations['area']<COCO_KP_MIN_AREA:
            continue


        OK_NR = 0
        l_keypoints_x = []
        l_keypoints_y = []
        for i in range(COCO_KP_NR):
            if l_key_points[i,2]>=COCO_KP_LEVEL:
                l_keypoints_x.append(l_key_points[i,0]/(image_width-1))
                l_keypoints_y.append(l_key_points[i,1]/(image_height-1))
                OK_NR += 1
            else:
                l_keypoints_x.append(-1)
                l_keypoints_y.append(-1)

        if OK_NR<COCO_KP_MIN_NR:
            continue

        keypoints_x.extend(l_keypoints_x)
        keypoints_y.extend(l_keypoints_y)

        has_keypoints = True

        xmin.append(float(x) / image_width)
        xmax.append(float(x + width) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(y + height) / image_height)
        is_crowd.append(object_annotations['iscrowd'])
        category_ids.append(category_id)
        category_names.append(category_index[reverse_trans_id(category_id)]['name'].encode('utf8'))
        area.append(object_annotations['area'])

        if include_masks:
            run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                                image_height, image_width)
            binary_mask = mask.decode(run_len_encoding)
            if not object_annotations['iscrowd']:
                binary_mask = np.amax(binary_mask, axis=2)
            pil_image = PIL.Image.fromarray(binary_mask)
            output_io = io.BytesIO()
            pil_image.save(output_io, format='PNG')
            encoded_mask_png.append(output_io.getvalue())

    if not has_keypoints:
        return None,None,None,None

    assert len(keypoints_x)==len(keypoints_y),"Error keypoints lens0 {len(keypoints_x)}, {len(keypoints_y)}."
    assert len(keypoints_x) == COCO_KP_NR*len(category_ids),"Error keypoints lens1 {len(keypoints_x}, {len(category_ids)}"
    #print("Xy")
    #wmlu.show_nparray(np.array(keypoints_x))
    #wmlu.show_nparray(np.array(keypoints_y))
    feature_dict = {
        'image/height':
            dataset_util.int64_feature(image_height),
        'image/width':
            dataset_util.int64_feature(image_width),
        'image/filename':
            dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':
            dataset_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            dataset_util.bytes_feature(key.encode('utf8')),
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
        'image/object/keypoints/x':
            dataset_util.float_list_feature(keypoints_x),
        'image/object/keypoints/y':
            dataset_util.float_list_feature(keypoints_y),
        'image/object/class/label':
            dataset_util.int64_list_feature(category_ids),
        'image/object/is_crowd':
            dataset_util.int64_list_feature(is_crowd),
        'image/object/area':
            dataset_util.float_list_feature(area),
        'image/file_index': dataset_util.int64_feature(src_file_index),
    }
    if include_masks:
        feature_dict['image/object/mask'] = (
            dataset_util.bytes_list_feature(encoded_mask_png))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    # category_id is the key of ID_TO_TEXT
    if len(category_ids) == 0:
        return None, None, None, None
    src_file_index += 1
    return key, example, num_annotations_skipped, repeat_nr


def _create_tf_record_from_coco_annotations(
        annotations_file, image_dir, output_path, include_masks, is_train_data=True, img_filter=None,
        name="coco_train"):
    """Loads COCO annotation json files and converts to tf.Record format.

    Args:
      annotations_file: JSON file containing bounding box annotations.
      image_dir: Directory containing the image files.
      output_path: Path to output tf.Record file.
      include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
    """
    if tf.gfile.IsDirectory(output_path):
        print(f"Remove {output_path}")
        shutil.rmtree(output_path)
    if not tf.gfile.IsDirectory(output_path):
        tf.gfile.MakeDirs(output_path)
    with tf.gfile.GFile(annotations_file, 'r') as fid:
        groundtruth_data = json.load(fid)
        images = groundtruth_data['images']
        if img_filter is not None:
            images = list(filter(img_filter, images))
            print(f"Image len {len(images)}.")
        category_index = label_map_util.create_category_index(
            groundtruth_data['categories'])

        annotations_index = {}
        if 'annotations' in groundtruth_data:
            tf.logging.info(
                'Found groundtruth annotations. Building annotations index.')
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
        tf.logging.info('%d images are missing annotations.',
                        missing_annotation_count)

        tf.logging.info('writing to output path: %s', output_path)

        total_num_annotations_skipped = 0
        fidx = 0
        idx = 0
        while True:
            fidx += 1
            save_path = get_save_path(output_path, name, fidx)
            if idx % 100 == 0:
                tf.logging.info('On image %d of %d', idx, len(images))
                if is_train_data and (TRAIN_SIZE_LIMIT is not None) and (idx > TRAIN_SIZE_LIMIT):
                    break
                elif (not is_train_data) and (VAL_SIZE_LIMIT is not None) and (idx > VAL_SIZE_LIMIT):
                    break
            with tf.python_io.TFRecordWriter(save_path) as writer:
                j = 0
                while j < IMAGE_PER_RECORD:
                    image = images[idx]
                    annotations_list = annotations_index[image['id']]
                    _, tf_example, num_annotations_skipped, repeat_nr = create_tf_example(
                        image, annotations_list, image_dir, category_index, include_masks)
                    if tf_example is not None:
                        total_num_annotations_skipped += num_annotations_skipped
                        assert repeat_nr > 0, f"Error repeat nr {repeat_nr}"
                        for i in range(repeat_nr):
                            writer.write(tf_example.SerializeToString())
                            j += 1
                    idx += 1
                    if idx >= len(images):
                        break

            if idx >= len(images):
                break
        tf.logging.info('Finished writing, skipped %d annotations.',
                        total_num_annotations_skipped)


def get_save_path(save_dir, name, fidx):
    return os.path.join(save_dir, f"{name}_{fidx:4}.tfrecord")


def main(_):
    SCRATCH_DIR = FLAGS.data_dir
    train_image_dir = os.path.join(SCRATCH_DIR, "train2017")
    val_image_dir = os.path.join(SCRATCH_DIR, "val2017")
    train_annotations_file = os.path.join(SCRATCH_DIR, "annotations/person_keypoints_train2017.json")
    val_annotations_file = os.path.join(SCRATCH_DIR, "annotations/person_keypoints_val2017.json")
    output_dir = SCRATCH_DIR

    train_output_path = os.path.join(output_dir,"tfdata_2017_kp_train")
    val_output_path = os.path.join(output_dir,"tfdata_2017_kp_val")
    if not tf.gfile.IsDirectory(output_dir):
        tf.gfile.MakeDirs(output_dir)

    _create_tf_record_from_coco_annotations(
        train_annotations_file,
        train_image_dir,
        train_output_path,
        FLAGS.include_masks,
        name='coco_train',
        is_train_data=True)

    _create_tf_record_from_coco_annotations(
        val_annotations_file,
        val_image_dir,
        val_output_path,
        FLAGS.include_masks,
        name='coco_val',
        is_train_data=False)


if __name__ == '__main__':
    tf.app.run()
