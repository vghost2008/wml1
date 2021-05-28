#coding=utf-8
import tensorflow as tf
from object_detection2.data.dataloader import DataLoader
import time
import tensorflow as tf
import wml_tfutils as wmlt
import wsummary
import wml_utils as wmlu
import iotoolkit.coco_tf_decodev2 as cocodecode
import iotoolkit.coco_toolkit as cocot
import wsummary
import iotoolkit.transform as trans
from object_detection2.config.config import CfgNode as CN
from object_detection2.data.datasets.build import DATASETS_REGISTRY
from object_detection2.data.dataloader import *
from wml_utils import  AvgTimeThis
from object_detection2.standard_names import *
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

tf.app.flags.DEFINE_string('logdir', wmlu.home_dir("ai/tmp/tools_logdir/"),"Logdir path")

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim
from object_detection2.data.buildin_dataprocess import DATAPROCESS_REGISTRY

aa = trans.RandomSelectSubTransform([
    [trans.WRandomTranslate(prob=1,pixels=20),trans.WRandomTranslate(prob=0.5,pixels=20)],
    [trans.WRandomTranslate(prob=1, pixels=20,translate_horizontal=False), trans.WRandomTranslate(prob=0.5, pixels=20,translate_horizontal=False)]
])

@DATAPROCESS_REGISTRY.register()
def simple_semantic1(cfg, is_training):
    if is_training:
        trans_on_single_img = [trans.MaskNHW2HWN(),
                                    trans.ResizeToFixedSize(),
                                    trans.MaskHWN2NHW(),
                                    trans.WRemoveCrowdInstance(False),
                                    trans.GetSemanticMaskFromCOCO(num_classes=3,no_background=False),
                                    trans.AddBoxLens(),
                                    trans.UpdateHeightWidth(),
                                    ]
        trans_on_batch_img = [trans.FixDataInfo()]
    else:
        trans_on_single_img = [trans.MaskNHW2HWN(),
                               trans.ResizeToFixedSize(),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.AddBoxLens(),
                               ]
        trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                              trans.FixDataInfo()]
    return trans_on_single_img,trans_on_batch_img

@DATAPROCESS_REGISTRY.register()
def test(cfg,is_training):
    return [
            trans.ShowInfo(),
            trans.BBoxesRelativeToAbsolute(),
             trans.RandomRotateAnyAngle(use_mask=False,rotate_bboxes_type=0,rotate_probability=1,max_angle=15),
             trans.RemoveZeroAreaBBox(),
             trans.GetSemanticMaskFromCOCO(num_classes=3,no_background=False),
             trans.UpdateHeightWidth(),
            trans.AddBoxLens(),
            ],\
           [trans.NoTransform(),trans.BBoxesAbsoluteToRelative()]
_C = CN()
_C.INPUT = CN()
_C.INPUT.DATAPROCESS = "NONE"
_C.INPUT.MIN_SIZE_TRAIN = (224,224)
_C.INPUT.MAX_SIZE_TRAIN = 1333
_C.INPUT.SIZE_ALIGN = 1
_C.DATASETS = CN()
_C.DATASETS.SKIP_CROWD_DURING_TRAINING = True
_C.DATASETS.TRAIN = "mot_train"
_C.SOLVER = CN()
_C.SOLVER.IMS_PER_BATCH = 1
_C.INPUT.STITCH = 0.0
_C.INPUT.ROTATE_ANY_ANGLE = CN()
_C.INPUT.ROTATE_ANY_ANGLE.ENABLE = True
_C.INPUT.ROTATE_ANY_ANGLE.MAX_ANGLE = 6
_C.INPUT.ROTATE_ANY_ANGLE.PROBABILITY = 0.5
_C.INPUT.FILTER_EMPTY = True
_C.INPUT.SHUFFLE_BUFFER_SIZE = 10
_C.INPUT.EXTRA_FILTER = ""
def main(_):
    cfg = _C
    data_loader = DataLoader(cfg=cfg,is_training=False)
    data_args = DATASETS_REGISTRY[cfg.DATASETS.TRAIN]
    with tf.device(":/cpu:0"):
        data,num_classes = data_loader.load_data(*data_args)
    res = data.get_next()
    print(res)
    data_loader.detection_image_summary(res,max_boxes_to_draw=200,max_outputs=4)
    wsummary.image_summaries(res['image'],'image')
    res[IMAGE] = tf.ones_like(res[IMAGE])*200
    data_loader.detection_image_summary(res,max_boxes_to_draw=200,max_outputs=4)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
    with tf.control_dependencies([tf.group(wmlt.get_hash_table_collection())]):
        initializer = tf.global_variables_initializer()
    sess.run(initializer)
    log_step = 1
    step = 0
    tt = AvgTimeThis()
    att = AvgTimeThis()
    while True:
        if step%log_step == 0:
            summary,bboxes,lens = sess.run([merged,res[GT_BOXES],res[GT_LENGTH]])
            wmlu.show_nparray(bboxes[0][:lens[0]])
            summary_writer.add_summary(summary, step)
        else:
            with tt:
                with att:
                    image,bbox,mask = sess.run([res[IMAGE],res[GT_MASKS],res[GT_BOXES]])
        print("step %5d, %f %f secs/batch" % (step,tt.get_time(),att.get_time()))
        step += 1
    sess.close()
    summary_writer.close()

if __name__ == "__main__":
    if tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.DeleteRecursively(FLAGS.logdir)
    tf.app.run()
