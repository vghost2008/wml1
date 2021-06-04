from object_detection2.modeling.backbone import *
from object_detection2.modeling.backbone.dla import build_any_dla_backbone
from object_detection2.config.config import get_cfg
import tensorflow as tf
from object_detection2.modeling.backbone.mobilenets import *
import wml_utils as wmlu
import wmodule


global_cfg = get_cfg()
#global_cfg.MODEL.MOBILENETS.MINOR_VERSION = "SMALL"
global_cfg.MODEL.DLA.BACKBONE = "build_resnet_backbone"
global_cfg.MODEL.RESNETS.DEPTH = 34
#global_cfg.MODEL.MOBILENETS.MINOR_VERSION = "LARGE"
net = tf.placeholder(tf.float32,[2,512,512,3])
x = {'image':net}
parent = wmodule.WRootModule()
mn = build_any_dla_backbone(global_cfg,parent=parent)
res = mn(x)

sess = tf.Session()
summary_writer = tf.summary.FileWriter(wmlu.home_dir("ai/tmp/tools_log"), sess.graph)

