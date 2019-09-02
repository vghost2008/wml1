import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops

def safe_divide(numerator, denominator, name):
    return tf.where(
        math_ops.greater(denominator, 0),
        math_ops.divide(numerator, denominator),
        tf.zeros_like(numerator),
        name=name)

def npsafe_divide(numerator, denominator, name):
    return np.where(
        np.greater(denominator, 0),
        np.divide(numerator, denominator),
        np.zeros_like(numerator))
