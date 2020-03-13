#coding=utf-8
import tensorflow as tf
from object_detection2.standard_names import *
def get_img_size_from_batched_inputs(inputs):
    image = inputs[IMAGE]
    return tf.shape(image)[1:3]