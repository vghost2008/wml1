import tensorflow as tf
from object_detection2.modeling.keypoints_heads.hrnet_pe_outputs import *
from object_detection2.config.config import *
import wmodule
import numpy as np

tf.enable_eager_execution()

cfg = get_cfg()
data = list(range(60))
#net = np.zeros([1,5,4,3],dtype=np.float32)
data = np.array(data,dtype=np.float32)
data = np.reshape(data,[1,5,4,3])
net = tf.convert_to_tensor(data)
#net = np.zeros([1,5,4,3],dtype=np.float32)
indexs = np.zeros([1,2,3],dtype=np.int32)
indexs[0,0,0] = 0
indexs[0,0,1] = 1
indexs[0,0,2] = 2

indexs[0,1,0] = 0+3
indexs[0,1,1] = 5+3
indexs[0,1,2] = 10+3

indexs = tf.convert_to_tensor(indexs)

length = np.array([2],dtype=np.int32)
length = tf.convert_to_tensor(length)

parent = wmodule.WRootModule()
output = HRNetPEOutputs(cfg.MODEL.KEYPOINTS,parent,gt_length=length,pred_maps=None)
loss = output.ae_loss(net,indexs)
print(loss)
