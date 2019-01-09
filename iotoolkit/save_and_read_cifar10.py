#!/usr/bin/python
import os
import sys
import numpy as np
import tensorflow as tf
import wmltools as wt
CIFAR_IMG_SIZE=30
def tf_int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def tf_byte_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
class DataSet:
	def __init__(self):
		self.images_data=np.array([])
		self.labels_data =np.array([])
	def setData(self,data):
		self.images_data = data[0]
		self.labels_data = data[1]

def read_raw_data(file_path):

	width          =   32
	height         =   32
	depth          =   3
	label_bytes    =   1
	image_bytes    =   width*height*depth
	record_bytes   =   label_bytes+image_bytes
	num_records    =   10000
	labels         =   []
	images         =   []

	with open(file_path,"rb") as fin:
		for _ in xrange(num_records):
			datas = np.frombuffer(fin.read(record_bytes),dtype=np.uint8)
			datas = datas.reshape(record_bytes)
			labels += [datas[0:1]]
			images += [datas[1:]]
	return labels,images,num_records
def convert_traindata_to_records(data_dir,save_dir):
	filenames = [ os.path.join(data_dir,'data_batch_%d.bin'%i) for i in range(1,6)]
	convert_to_records(filenames,os.path.join(save_dir,"traindata.tfrecords"))
def convert_testdata_to_records(data_dir,save_dir):
	filenames = [os.path.join(data_dir,'test_batch.bin')]
	convert_to_records(filenames,os.path.join(save_dir,"testdata.tfrecords"))
def convert_to_records(datapaths,savepath):
	width          =   32
	height         =   32
	depth          =   3
	writer = tf.python_io.TFRecordWriter(savepath)
	for datapath in datapaths:
		print("Trans file ",datapath)
		labels,images,num_records = read_raw_data(datapath)
		for i in xrange(num_records):
			label = wt.to_one_hot(labels[i])
			image = images[i]
			image = image.reshape(depth,height,width)
			image = image.transpose(1,2,0)
			label_raw = label.tostring()
			image_raw = image.tostring()
			example = tf.train.Example(features=tf.train.Features(feature={
				'height':tf_int64_feature(32),
				'width':tf_int64_feature(32),
				'label':tf_byte_feature(label_raw),
				'image_raw':tf_byte_feature(image_raw)}))
			writer.write(example.SerializeToString())
	writer.close
def to_records(data_dir,save_dir):
	if not tf.gfile.Exists(save_dir):
		tf.gfile.MakeDirs(save_dir)
	convert_traindata_to_records(data_dir,save_dir)
	convert_testdata_to_records(data_dir,save_dir)
def read_and_decode(filename_queue):
	with tf.name_scope("read_and_decode"):
		reader = tf.TFRecordReader()
		_,serialized_example = reader.read(filename_queue)
		features = tf.parse_single_example(serialized_example,features={'image_raw':tf.FixedLenFeature([],tf.string),
		'label':tf.FixedLenFeature([],tf.string)})

		image = tf.decode_raw(features['image_raw'],tf.uint8)
		label = tf.decode_raw(features['label'],tf.int64)
		with tf.name_scope("reshape_and_cast"):
			label.set_shape([10])
			label = tf.cast(label,tf.float32)
			image.set_shape([32*32*3])
			image = tf.cast(image,tf.float32)
			image = tf.reshape(image,[32,32,3]);
		with tf.name_scope("image_adjust"):
			image = tf.random_crop(image,[CIFAR_IMG_SIZE,CIFAR_IMG_SIZE,3])
			image = tf.image.random_flip_left_right(image)
			image = tf.image.random_brightness(image,max_delta=63)
			image = tf.image.random_contrast(image,lower=0.2,upper=1.8)
			image = (image-127.5)*2./255.
	return image,label
def eval_read_and_decode(filename_queue):
	with tf.name_scope("read_and_decode"):
		reader = tf.TFRecordReader()
		_,serialized_example = reader.read(filename_queue)
		features = tf.parse_single_example(serialized_example,features={'image_raw':tf.FixedLenFeature([],tf.string),
		'label':tf.FixedLenFeature([],tf.string)})

		image = tf.decode_raw(features['image_raw'],tf.uint8)
		label = tf.decode_raw(features['label'],tf.int64)
		with tf.name_scope("reshape_and_cast"):
			label.set_shape([10])
			label = tf.cast(label,tf.float32)
			image.set_shape([32*32*3])
			image = tf.cast(image,tf.float32)
			image = tf.reshape(image,[32,32,3]);
			image = tf.random_crop(image,[CIFAR_IMG_SIZE,CIFAR_IMG_SIZE,3])
			image = (image-127.5)*2./255.
	return image,label
