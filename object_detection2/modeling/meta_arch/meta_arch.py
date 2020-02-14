#coding=utf-8
import wmodule
from object_detection2.standard_names import *
import tensorflow as tf

class MetaArch(wmodule.WModule):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        with tf.name_scope("preprocess_image"):
            batched_inputs[IMAGE] = (batched_inputs[IMAGE]-127.5)/127.5
            return batched_inputs