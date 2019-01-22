import tensorflow as tf
from tensorflow.python.ops import math_ops

def safe_divide(numerator, denominator, name):
    return tf.where(
        math_ops.greater(denominator, 0),
        math_ops.divide(numerator, denominator),
        tf.zeros_like(numerator),
        name=name)
