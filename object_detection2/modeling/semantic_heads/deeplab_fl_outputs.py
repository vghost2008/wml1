import tensorflow as tf
import wmodule
import basic_tftools as btf
from object_detection2.standard_names import *
from object_detection2.modeling.build import HEAD_OUTPUTS
import wnn


@HEAD_OUTPUTS.register()
class DeepLabFLOutputs(wmodule.WChildModule):
    def __init__(self, pred_logits, labels, cfg, parent, *args, **kwargs):
        super().__init__(cfg, parent=parent, *args, **kwargs)
        self.labels = labels
        self.logits = pred_logits
        self.upsample_logits = self.cfg.UPSAMPLE_LOGITS  # True
        assert cfg.PRED_BACKGROUND==False,"Error pred background value."
        self.num_classes = cfg.NUM_CLASSES  # add background

    def losses(self):
        with tf.name_scope("deeplab_loss"):
            if self.upsample_logits:
                logits = tf.image.resize_bilinear(self.logits,
                                                  btf.img_size(self.labels),
                                                  align_corners=True)
                labels = self.labels
            else:
                logits = self.logits
                labels = tf.image.resize_nearest_neighbor(self.labels,
                                                          btf.img_size(self.logits),
                                                          align_corners=True)
            C = btf.channel(labels)
            labels = tf.reshape(labels, shape=[-1, C])
            labels = tf.cast(labels, tf.float32)
            labels = labels[...,1:]
            logits = tf.reshape(logits, [-1, self.num_classes])
            loss = wnn.sigmoid_cross_entropy_with_logits_FL(labels=labels,
                                                            logits=logits, 
                                                            gamma=self.cfg.FOCAL_LOSS_GAMMA,
                                                            alpha=self.cfg.FOCAL_LOSS_ALPHA)
            loss = tf.reduce_sum(loss)
            return {"semantic_loss": loss}

    @btf.add_name_scope
    def inference(self, inputs, logits):
        size = btf.img_size(inputs[IMAGE])
        logits = tf.image.resize_bilinear(logits, size, align_corners=True)
        shape = btf.combined_static_and_dynamic_shape(logits)
        shape[-1] = 1
        background = tf.ones(shape=shape,dtype=tf.float32)*self.cfg.SCORE_THRESH_TEST
        probs = tf.nn.sigmoid(logits)
        probs = tf.concat([background,probs],axis=-1)
        mask = tf.argmax(probs, 3)+1
        semantic = tf.one_hot(mask, depth=self.num_classes+1, on_value=1.0, off_value=0.0)
        return {RD_SPARSE_SEMANTIC: mask,
                RD_SEMANTIC: semantic}

