import tensorflow as tf
import basic_tftools as btf

def boxes_nms(bboxes, classes, threshold=0.45,confidence=None,classes_wise=True,k=-1,max_coord_value=1.0):
    if classes_wise:
        if max_coord_value is None:
            max_coord_value = tf.reduce_max(bboxes)
        max_coord_value += 0.1
        offset = tf.ones_like(bboxes)*tf.expand_dims(tf.cast(classes,tf.float32),axis=-1)*max_coord_value
        tbboxes = bboxes+offset
    else:
        tbboxes = bboxes
    if k is None or k <= 0:
        k = int(1e10)
    if confidence is None:
        box_nr = btf.batch_size(bboxes)
        data = tf.range(box_nr)
        data = tf.to_float(data)
        confidence = tf.reverse(data,axis=[0])/tf.to_float(box_nr)
    indices = tf.image.non_max_suppression(tbboxes,scores=confidence,
                                           iou_threshold=threshold,
                                           max_output_size=k)
    r_bboxes = tf.gather(bboxes,indices)
    r_labels = tf.gather(classes,indices)
    return r_bboxes,r_labels,indices
