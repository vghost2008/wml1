#coding=utf-8
import os
import sys
import random
import time
import numpy as np
import tensorflow as tf
from iotoolkit.pascal_voc_toolkit import read_voc_xml
import shutil
import xml.etree.ElementTree as ET
import sys
from iotoolkit.pascal_voc_data import *
from wml_tfutils import int64_feature,bytes_feature,floats_feature
from iotoolkit.pascal_voc_data import TEXT_TO_ID

DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'
SAMPLES_PER_FILES = 6000
'''def category_id_filter(category_id):
    good_ids = [15,6,7,14,2]
    return category_id in good_ids

def labels_text_to_labels(labels_text):
    return [TEXT_TO_ID[x] for x in labels_text]'''
'''text = []
for i in range(ord('a'),ord('z')+1):
    text.append(chr(i))
for i in range(ord('A'), ord('Z') + 1):
    text.append(chr(i))
for i in range(ord('0'), ord('9') + 1):
    text.append(chr(i))
text.append('/')
text.append('\\')
text.append('-')
text.append('+')
text.append(":")
text.append("WORD")
text_to_id = {}
for i,t in enumerate(text):
    text_to_id[t] = i+1'''

class VOCMaker(object):
    def __init__(self,filenames=None):
        if filenames is not None and isinstance(filenames,str):
            with open(filenames) as f:
                self.filenames = [x.strip() for x in f.readlines()]
        else:
            self.filenames = filenames

        def _category_id_filter(category_id):
            return True

        self.category_id_filter = _category_id_filter

        def _labels_text_to_labels(labels_text):
            # return [int(x) for x in labels_text]
            for x in labels_text:
                if x not in TEXT_TO_ID:
                    print(f"Error \"{x}\" not in target set.")
            return [TEXT_TO_ID[x] for x in labels_text]

        self.labels_text_to_labels = _labels_text_to_labels

    '''
    directory:图像目录路径
    name:图像文件名，但不包含路径及文件名
    
    返回图像数据，bbox(用[0,1]表示，bbox相对应的label
    '''
    def _process_image(self,directory, name):
        filename = os.path.join(directory,DIRECTORY_IMAGES + name + '.jpg')
        if not os.path.exists(filename):
            return None,None,None,None,None,None,None
        image_data = tf.gfile.FastGFile(filename, 'rb').read()

        filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
        shape, _bboxes, _labels_text, _difficult, _truncated = read_voc_xml(filename, adjust=None)
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
    def _add_to_tfrecord(self,dataset_dir, name, tfrecord_writer):
        if self.filenames is not None and name not in self.filenames:
            return False
        image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
            self._process_image(dataset_dir, name)
        if image_data is None:
            return False
        example = self._convert_to_example(image_data, labels, labels_text,
                                      bboxes, shape, difficult, truncated)
        tfrecord_writer.write(example.SerializeToString())
        return True


    def _get_output_filename(self,output_dir, name, idx):
        return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)

    '''
    将所有图像文件按SAMPLES_PER_FILES(200)一批保存在tfrecored文件中
    '''
    def to_tfrecords(self,dataset_dir, output_dir, name='train', shuffling=False,repeat=1,fidx=0):
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
        if repeat>1:
            filenames = list(filenames)*repeat
        if shuffling:
            random.seed(time.time())
            random.shuffle(filenames)

        i = 0
        while i < len(filenames):
            tf_filename = self._get_output_filename(output_dir, name, fidx)
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                j = 0
                while i < len(filenames) and j < SAMPLES_PER_FILES:
                    sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                    sys.stdout.flush()

                    filename = filenames[i]
                    img_name = filename[:-4]
                    if self._add_to_tfrecord(dataset_dir, img_name, tfrecord_writer):
                        j += 1
                    i += 1

                fidx += 1
        print('\nFinished converting the dataset total %d examples.!'%(len(filenames)))

if __name__ == "__main__":

    dataset_dir = "/media/vghost/Linux/constantData/MachineLearning/mldata/PASCAL/VOCdevkit/VOC2012"
    output_dir = "/home/vghost/ai/mldata/VOC2012_tfdata"
    #dataset_dir = "/home/vghost/ai/mldata/ocrdatav1/rdatav2"
    #output_dir = "/home/vghost/ai/mldata/ocrdatav1/tfdata1"
    output_name = "train"

    print('Dataset directory:', dataset_dir)
    print('Output directory:',output_dir)
    m = VOCMaker()
    m.to_tfrecords(dataset_dir, output_dir, output_name)