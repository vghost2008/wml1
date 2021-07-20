import glob
import random
import time
import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np

#import maskrcnn_benchmark.layers.nms as nms
# Set printoptions
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)


def float3(x):  # format floats to 3 decimals
    return float(format(x, '.3f'))


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
