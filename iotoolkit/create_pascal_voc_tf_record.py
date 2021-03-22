#coding=utf-8
import os
import functools
import sys
import random
import time
import numpy as np
from multiprocessing import Pool
import tensorflow as tf
import object_detection.utils as odu
import object_detection.npod_toolkit as npod
import shutil
import xml.etree.ElementTree as ET
import sys
from iotoolkit.pascal_voc_data import *
from wml_tfutils import int64_feature,bytes_feature,floats_feature
from iotoolkit.pascal_voc_data import TEXT_TO_ID
import img_utils as wmli
import wml_utils as wmlu

SAMPLES_PER_FILES = 6000
def _category_id_filter(category_id):
    return True

def _labels_text_to_labels(labels_text):
    for x in labels_text:
        if x not in TEXT_TO_ID:
           print(f"Error \"{x}\" not in target set.")
    return [TEXT_TO_ID[x] for x in labels_text]

class VOCMaker(object):
    def __init__(self,filenames=None):
        if filenames is not None and isinstance(filenames,str):
            with open(filenames) as f:
                self.filenames = [x.strip() for x in f.readlines()]
        else:
            self.filenames = filenames


        self.category_id_filter = _category_id_filter
        self.image_preprocess = None
        #输入为list(str)
        self.labels_text_to_labels = _labels_text_to_labels

    '''
    directory:图像目录路径
    name:图像文件名，但不包含路径及文件名
    
    返回图像数据，bbox(用[0,1]表示，bbox相对应的label
    '''
    def _process_image(self,xml_file,img_file):
        if not os.path.exists(img_file):
            return None,None,None,None,None,None,None
        if self.image_preprocess is not None:
            img = wmli.imread(img_file)
            img = self.image_preprocess(img)
            image_data = wmli.encode_img(img)
        else:
            image_data = tf.gfile.FastGFile(img_file, 'rb').read()

        shape, _bboxes, _labels_text, _difficult, _truncated,_ = odu.read_voc_xml(xml_file, adjust=None)
        _labels = self.labels_text_to_labels(_labels_text)
        bboxes = []
        labels_text = []
        difficult = []
        truncated = []
        labels = []
        for data in zip(_bboxes,_labels,_labels_text,_difficult,_truncated):
            if self.category_id_filter(data[1]):
                bboxes.append(data[0])
                labels.append(data[1])
                labels_text.append(data[2])
                difficult.append(data[3])
                truncated.append(data[4])

        if len(labels) == 0:
            #print(f"Ignore {name}.")
            return None,None,None,None,None,None,None
        return image_data, shape, bboxes, labels, labels_text, difficult, truncated


    def _convert_to_example(self,image_data, labels, labels_text, bboxes, shape,
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
            'image/object/bbox/xmin': floats_feature(xmin),
            'image/object/bbox/xmax': floats_feature(xmax),
            'image/object/bbox/ymin': floats_feature(ymin),
            'image/object/bbox/ymax': floats_feature(ymax),
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
    def _add_to_tfrecord(self,img_file, tfrecord_writer):
        xml_file = wmlu.change_suffix(img_file,"xml")
        if not os.path.exists(img_file) or not os.path.exists(xml_file):
            print(f"Error file {xml_file}, {img_file}.")
            return False
        image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
            self._process_image(xml_file,img_file)
        if image_data is None:
            return False
        example = self._convert_to_example(image_data, labels, labels_text,
                                      bboxes, shape, difficult, truncated)
        tfrecord_writer.write(example.SerializeToString())
        return True


    def _get_output_filename(self,output_dir, name, idx):
        return '%s/%s_%04d.tfrecord' % (output_dir, name, idx)

    def make_tfrecord(self,file_data,output_dir,name="train"):
        fidx,files = file_data
        tf_filename = self._get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            for file in files:
                self._add_to_tfrecord(file, tfrecord_writer)

    '''
    将所有图像文件按SAMPLES_PER_FILES(200)一批保存在tfrecored文件中
    '''
    def multi_thread_to_tfrecords(self,dataset_dir, output_dir, shuffling=False,fidx=0):
        files = wmlu.recurse_get_filepath_in_dir(dataset_dir,suffix=".jpg")

        return self.multi_thread_to_tfrecords_by_files(files,output_dir,shuffling,fidx)
    '''
    将所有图像文件按SAMPLES_PER_FILES(200)一批保存在tfrecored文件中
    files: img file list
    '''
    def multi_thread_to_tfrecords_by_files(self,files, output_dir,shuffling=False,fidx=0):
        wmlu.create_empty_dir(output_dir,remove_if_exists=True,yes_to_all=True)
        if shuffling:
            random.seed(time.time())
            random.shuffle(files)
        wmlu.show_list(files[:100])
        if len(files)>100:
            print("...")
        print(f"Total {len(files)} files.")
        sys.stdout.flush()
        files = wmlu.list_to_2dlist(files,SAMPLES_PER_FILES)
        files_data = list(enumerate(files))
        if fidx != 0:
            _files_data = []
            for fid,file_d in files_data:
                _files_data.append([fid+fidx,file_d])
            files_data = _files_data
        sys.stdout.flush()
        pool = Pool(13)
        pool.map(functools.partial(self.make_tfrecord,output_dir=output_dir),files_data)
        #list(map(functools.partial(self.make_tfrecord,output_dir=output_dir),files_data))
        pool.close()
        pool.join()

        print('\nFinished converting the dataset total %d examples.!'%(len(files)))

if __name__ == "__main__":

    dataset_dir = "/media/vghost/Linux/constantData/MachineLearning/mldata/PASCAL/VOCdevkit/VOC2012"
    output_dir = "/home/vghost/ai/mldata/VOC2012_tfdata"
    output_name = "train"

    print('Dataset directory:', dataset_dir)
    print('Output directory:',output_dir)
    m = VOCMaker()
    m.to_tfrecords(dataset_dir, output_dir, output_name)
