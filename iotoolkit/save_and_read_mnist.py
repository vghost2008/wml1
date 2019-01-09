#!/usr/bin/python
import os
import sys
import numpy as np
import tensorflow as tf
import sys
sys.path.append("..")
import wml_tfutils as wmlt

slim = tf.contrib.slim

TRAIN_IMAGES = 'train-images.idx3-ubyte'
TRAIN_LABELS = 'train-labels.idx1-ubyte'
TEST_IMAGES = 't10k-images.idx3-ubyte'
TEST_LABELS = 't10k-labels.idx1-ubyte'

def read_mnist_32(bytestream):
	dt = np.dtype(np.uint32).newbyteorder('>')
	return np.frombuffer(bytestream.read(4),dtype=dt)[0]

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

def read_images_data(file_path):
	with open(file_path,"rb") as fin:
		magic = read_mnist_32(fin)
		if magic != 2051:
			raise ValueError("Invalid magic number %d in MNIST image file:%s"%(magic,fin.name))
		num_images = read_mnist_32(fin)
		num_rows = read_mnist_32(fin)
		num_columns = read_mnist_32(fin)
		data = np.frombuffer(fin.read(num_rows*num_columns*num_images),dtype=np.uint8)
		data = data.reshape(num_images,num_rows,num_columns,1)
		return data

def read_labels_data(file_path):
	with open(file_path,"rb") as fin:
		magic = read_mnist_32(fin)
		if magic != 2049:
				raise ValueError("Invalid magic number %d in MNIST labels file:%s"%(magic,fin.name))
		num_items = read_mnist_32(fin)
		data = np.frombuffer(fin.read(num_items),dtype=np.uint8)
		data = data.reshape(num_items)
		return data

def read_data(data_dir):
	train_data = DataSet()
	test_data = DataSet()
	train_images_data = read_images_data(os.path.join(data_dir,TRAIN_IMAGES))
	train_labels_data = read_labels_data(os.path.join(data_dir,TRAIN_LABELS))
	test_images_data = read_images_data(os.path.join(data_dir,TEST_IMAGES))
	test_labels_data = read_labels_data(os.path.join(data_dir,TEST_LABELS))

	train_data.setData((train_images_data,train_labels_data))
	test_data.setData((test_images_data,test_labels_data))

	return train_data,test_data
def convert_to_records(save_file_name,data_set):
	print("Writing ",save_file_name)
	images = data_set.images_data;
	labels = data_set.labels_data;

	writer = tf.python_io.TFRecordWriter(save_file_name)

	num = images.shape[0]
	rows = images.shape[1]
	cols = images.shape[2]
	for index in range(num):
		image_raw = images[index].tostring()
		l = labels[index]
		example = tf.train.Example(features=tf.train.Features(feature={
			'height':tf_int64_feature(rows),
			'width':tf_int64_feature(cols),
			'label':tf_int64_feature(l),
			'image_raw':tf_byte_feature(image_raw)}))
		writer.write(example.SerializeToString())
	writer.close

def to_records(data_dir,save_dir):
	datas = read_data(data_dir)
	if not tf.gfile.Exists(save_dir):
		tf.gfile.MakeDirs(save_dir)
	convert_to_records(os.path.join(save_dir,"train_data.tfrecord"),datas[0])
	convert_to_records(os.path.join(save_dir,"test_data.tfrecord"),datas[1])

def read_and_decode(filename_queue):
	reader = tf.TFRecordReader()
	_,serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,features={'image_raw':tf.FixedLenFeature([],tf.string),
	'label':tf.FixedLenFeature([],tf.string),
	'height':tf.FixedLenFeature([],tf.int64),
	'width':tf.FixedLenFeature([],tf.int64)})
	image = tf.decode_raw(features['image_raw'],tf.uint8)
	height = tf.cast(features['height'],tf.int32)
	width = tf.cast(features['width'],tf.int32)
	label = tf.decode_raw(features['label'],tf.int64)

	image.set_shape([784])
	label.set_shape([10])
	image = tf.reshape(image,[28,28,1]);
	image = tf.cast(image,tf.float32)*(1./255.)-0.5
	label = tf.cast(label,tf.float32)
	return image,label


def get_database(dataset_dir, split_name="train",
             file_pattern='%s_*.tfrecord',
          num_samples=1, items_to_descriptions=None, num_classes=10):

    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    reader = tf.TFRecordReader
    keys_to_features = {
        'image_raw': tf.FixedLenFeature([],dtype=tf.string),
        'label': tf.FixedLenFeature((), tf.int64),
        'format': tf.FixedLenFeature([], dtype=tf.string,default_value="raw"),

    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image_raw',"format",channels=1),
        'label': slim.tfexample_decoder.Tensor('label'),
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


def get_data(data_dir,num_samples=1,num_classes=10,batch_size=32):
    dataset = get_database(dataset_dir=data_dir,num_classes=num_classes,num_samples=num_samples)
    with tf.name_scope('data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=3,
            common_queue_capacity=20 * batch_size,
            common_queue_min=3*batch_size,
            shuffle=True)
        [image,label] = provider.get(["image","label"])
        image = tf.reshape(image,[28,28,1])
        image = tf.cast(image,tf.float32)
        wmlt.image_summaries(image, "raw_image", max_outputs=5)

        b_image, b_label  = tf.train.shuffle_batch([image, label],
                                                   batch_size=batch_size, num_threads=2,
                                                   capacity=5 * batch_size,
                                                   min_after_dequeue=2 * batch_size)

    return b_image,b_label