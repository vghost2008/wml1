# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys

from absl import logging
import numpy as np
import tensorflow as tf

def drop_connect(inputs, is_training, drop_connect_rate):
  """Apply drop connect."""
  if not is_training:
    return inputs

  # Compute keep_prob
  # TODO(tanmingxing): add support for training progress.
  keep_prob = 1.0 - drop_connect_rate

  # Compute drop_connect tensor
  batch_size = tf.shape(inputs)[0]
  random_tensor = keep_prob
  random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  output = tf.div(inputs, keep_prob) * binary_tensor
  return output



class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D, tf.layers.Layer):
  """Wrap keras DepthwiseConv2D to tf.layers."""

  pass


