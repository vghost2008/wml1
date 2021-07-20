from thirdparty.registry import Registry
import numpy as np
from iotoolkit.coco_toolkit import *
from object_detection2.standard_names import *

FILTER_REGISTRY = Registry("Filter")

def build_filter(name):
    return FILTER_REGISTRY.get(name)

@FILTER_REGISTRY.register()
def coco2017_balance_sample(x):
    threshold = 1.0
    labels = x[GT_LABELS]
    freq = tf.gather(COMPRESSED_ID_TO_FREQ,labels)
    print(COMPRESSED_ID_TO_FREQ)
    freq = tf.reduce_min(freq)
    v = threshold/(freq+1e-8)
    p = tf.random_uniform(shape=(),minval=0,maxval=1)
    return tf.less_equal(p,v)

@FILTER_REGISTRY.register()
def coco2014_balance_sample(x):
    threshold = 1.0
    labels = x[GT_LABELS]
    freq = tf.gather(ID_TO_FREQ,labels)
    print(ID_TO_FREQ)
    freq = tf.reduce_min(freq)
    v = threshold/(freq+1e-8)
    p = tf.random_uniform(shape=(),minval=0,maxval=1)
    return tf.less_equal(p,v)
