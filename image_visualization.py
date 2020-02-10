#coding=utf-8
import tensorflow as tf
import semantic.visualization_utils as smv

def __draw_detection_image_summary(images,
                           boxes,
                           classes=None,
                           scores=None,
                           category_index=None,
                           instance_masks=None,
                           keypoints=None,
                           max_boxes_to_draw=20,
                           min_score_thresh=0.2):
  """Draws bounding boxes, masks, and keypoints on batch of image tensors.

  Args:
    images: A 4D uint8 image tensor of shape [N, H, W, C].
    boxes: [N, max_detections, 4] float32 tensor of detection boxes.
    classes: [N, max_detections] int tensor of detection classes. Note that
      classes are 1-indexed.
    scores: None or [N, max_detections] float32 tensor of detection scores.
    category_index: a dict that maps integer ids to category dicts. e.g.
      {1: {'name': 'dog'}, 2: {'name': 'cat'}, ...}
    instance_masks: A 4D uint8 tensor of shape [N, max_detection, H, W] with
      instance masks.
    keypoints: A 4D float32 tensor of shape [N, max_detection, num_keypoints, 2]
      with keypoints.
    max_boxes_to_draw: Maximum number of boxes to draw on an image. Default 20.
    min_score_thresh: Minimum score threshold for visualization. Default 0.2.

  Returns:
    4D image tensor of type uint8, with boxes drawn on top.
  """
  if classes is None and scores is None  and instance_masks is None and keypoints is None:
      return tf.image.draw_bounding_boxes(images,boxes)

  if images.dtype == tf.float32 or images.dtype == tf.float64:
    min = tf.reduce_min(images)
    max = tf.reduce_max(images)
    images = (images-min)*255/(max-min+1e-8)
    images = tf.clip_by_value(images,0,255)
    images = tf.cast(images,tf.uint8)

  if images.get_shape().as_list()[-1] == 1:
      images = tf.tile(images,[1,1,1,3])

  if category_index is None:
    category_index = {}
    for i in range(100):
      category_index[i] = {'name':str(i)}

  if classes is None:
      classes = tf.ones(tf.shape(boxes)[:2],dtype=tf.int32)

  if scores is None:
      scores = tf.ones_like(classes,tf.float32)

  images = smv.draw_bounding_boxes_on_image_tensors(images,
                                         boxes,
                                         classes,
                                         scores,
                                         category_index,
                                         instance_masks,
                                         keypoints,
                                         max_boxes_to_draw,
                                         min_score_thresh)
  return images

def draw_detection_image_summary(images,
                                 boxes,
                                 classes=None,
                                 scores=None,
                                 category_index=None,
                                 instance_masks=None,
                                 keypoints=None,
                                 lengths=None,
                                 max_boxes_to_draw=20,
                                 min_score_thresh=0.2):
    if lengths is None:
        return __draw_detection_image_summary(images=images,boxes=boxes,
                                            classes=classes,
                                            scores=scores,
                                            category_index=category_index,
                                            instance_masks=instance_masks,keypoints=keypoints,
                                            max_boxes_to_draw=max_boxes_to_draw,
                                            min_score_thresh=min_score_thresh)
    else:
        if classes is None and scores is None  and instance_masks is None and keypoints is None:
            def fn(image,boxes,len):
                boxes = boxes[:len]
                image = tf.expand_dims(image,axis=0)
                boxes = tf.expand_dims(boxes,axis=0)
                image = tf.image.draw_bounding_boxes(image,boxes)
                return tf.squeeze(image,axis=0)
            images = tf.map_fn(lambda x:fn(x[0],x[1],x[2]),elems=[images,boxes,lengths],dtype=images.dtype)
            return images
        elif classes is not None and scores is None  and instance_masks is None and keypoints is None:
            def fn(image,boxes,classes,len):
                boxes = boxes[:len]
                classes = tf.expand_dims(classes[:len],axis=0)
                image = tf.expand_dims(image,axis=0)
                boxes = tf.expand_dims(boxes,axis=0)
                image = __draw_detection_image_summary(image,boxes,classes,scores,
                                         category_index,
                                         instance_masks,
                                         keypoints,
                                         max_boxes_to_draw,
                                         min_score_thresh)
                return tf.squeeze(image,axis=0)
            images = tf.map_fn(lambda x:fn(x[0],x[1],x[2],x[3]),elems=[images,boxes,classes,lengths],dtype=tf.uint8)
            return images
        elif classes is not None and scores is not None  and instance_masks is None and keypoints is None:
            def fn(image,boxes,classes,scores,len):
                boxes = boxes[:len]
                classes = classes[:len]
                scores = scores[:len]
                image = tf.expand_dims(image,axis=0)
                boxes = tf.expand_dims(boxes,axis=0)
                image = __draw_detection_image_summary(image,boxes,classes,scores,
                                                       category_index,
                                                       instance_masks,
                                                       keypoints,
                                                       max_boxes_to_draw,
                                                       min_score_thresh)
                return tf.squeeze(image,axis=0)
            images = tf.map_fn(lambda x:fn(x[0],x[1],x[2],x[3],x[4]),elems=[images,boxes,classes,scores,lengths],dtype=tf.uint8)
            return images
        else:
            #Need to do
            pass
