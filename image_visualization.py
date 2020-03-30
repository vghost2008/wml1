#coding=utf-8
import tensorflow as tf
import semantic.visualization_utils as smv
import basic_tftools as btf
import semantic.visualization_utils as visu
import numpy as np
import cv2

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
      if images.dtype == tf.float32 or images.dtype == tf.float64:
          min = tf.reduce_min(images)
          max = tf.reduce_max(images)
          images = (images-min)/(max-min+1e-8)
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

@btf.add_name_scope
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
    if instance_masks is not None:
        def fn(image,mask,len):
            mask = mask[:len]
            return draw_mask_on_image(image,mask)
        if lengths is None:
            batch_size,nr,H,W = btf.combined_static_and_dynamic_shape(instance_masks)
            lengths = tf.ones([batch_size],dtype=tf.int32)*nr
        if images.dtype is not tf.uint8:
            min = tf.reduce_min(images)
            max = tf.reduce_max(images)
            value_range = (max-min)
            images = tf.cond(tf.less(value_range,127.5),lambda:tf.cast(tf.clip_by_value((images+1.0)*127.5,0,255),tf.uint8),
                             lambda:tf.cast(images,tf.uint8))
        images = tf.map_fn(lambda x:fn(x[0],x[1],x[2]),elems=[images,instance_masks,lengths],dtype=tf.uint8)
        return draw_detection_image_summary(images,boxes,classes,scores=scores,
                                            category_index=category_index,
                                            instance_masks=None,
                                            keypoints=keypoints,
                                            lengths=lengths,
                                            max_boxes_to_draw=max_boxes_to_draw,
                                            min_score_thresh=min_score_thresh)
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
                old_type = None
                if image.dtype != tf.float32:
                    old_type = image.dtype
                    image = tf.cast(image,tf.float32)
                image = tf.image.draw_bounding_boxes(image,boxes)
                if old_type is not None:
                    image = tf.cast(image,old_type)
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
                classes = tf.expand_dims(classes,axis=0)
                scores = tf.expand_dims(scores,axis=0)
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
            raise NotImplementedError()


@btf.add_name_scope
def draw_positive_box_on_images(image,boxes,pmasks):
    '''

    :param image: [batch_size,H,W,C]
    :param boxes: [batch_size,box_nr,4]
    :param pmasks: [batch_size,box_nr] tf.bool
    :return: image with positive box [X,H,W,C]
    '''
    image = btf.static_or_dynamic_map_fn(lambda x:_draw_positive_box_on_single_images(x[0],x[1],x[2]),
                                         elems=[image,boxes,pmasks])
    return image


def _draw_positive_box_on_single_images(image,boxes,pmasks):
    '''

    :param image: [H,W,C]
    :param boxes: [box_nr,4]
    :param pmasks: [box_nr] tf.bool
    :return: image with positive box [H,W,C]
    '''
    boxes = tf.boolean_mask(boxes,pmasks)
    image = tf.expand_dims(image,axis=0)
    boxes = tf.expand_dims(boxes,axis=0)
    image = tf.image.draw_bounding_boxes(image,boxes)
    return tf.squeeze(image,axis=0)

'''
image:[H,W,C]
mask: [N,H,W]
color: [N]
'''
@btf.add_name_scope
def draw_mask_on_image(image, mask, color=None,alpha=0.4,no_first_mask=False,name='summary_image_with_mask'):
    with tf.device("/cpu:0"):
        if no_first_mask:
            mask = mask[:,:,1:]

        if color is None:
            with tf.device("/cpu:0"):
                mask_nr = tf.shape(mask)[2]
                color_nr = len(visu.MIN_RANDOM_STANDARD_COLORS)
                color_tensor = tf.convert_to_tensor(visu.MIN_RANDOM_STANDARD_COLORS)
                color = tf.gather(color_tensor,
                                  tf.mod(tf.range(mask_nr,dtype=tf.int32), color_nr))
        image = visu.tf_draw_masks_on_image(image=image,mask=mask,color=color,alpha=alpha)
        return image

'''
masks:shape=[batch_size,N,h,w]
boxes:shape=[batch_size,N,4]
size:[H,W]
mask_bg_value:mask background value
return:
shape=[batch_size,N,H,W]
'''
@btf.add_name_scope
def batch_tf_get_fullsize_mask(boxes,masks,size,mask_bg_value=0):
    return tf.map_fn(lambda x:tf_get_fullsize_mask(x[0],x[1],size,mask_bg_value),
                     elems=[boxes,masks],
                     dtype=masks.dtype)
'''
masks:shape=[N,h,w]
boxes:shape=[N,4]
size:[H,W]
return:
shape=[N,H,W]
'''
@btf.add_name_scope
def tf_get_fullsize_mask(boxes,masks,size,mask_bg_value=0):
    res = tf.py_func(get_fullsize_mask,[boxes,masks,size,mask_bg_value],Tout=masks.dtype)
    N,h,w = btf.combined_static_and_dynamic_shape(masks)
    H,W = size[0],size[1]
    return tf.reshape(res,[N,H,W])

'''
bboxes:[(ymin,xmin,ymax,xmax),....] value in range[0,1]
mask:[X,h,w]
size:[H,W]
'''
def get_fullsize_mask(boxes,masks,size,mask_bg_value=0):
    dtype = masks.dtype

    res_masks = []
    for i,bbox in enumerate(boxes):
        x = int(bbox[1]*size[1])
        y = int(bbox[0]*size[0])
        w = int((bbox[3]-bbox[1])*size[1])
        h = int((bbox[2]-bbox[0])*size[0])
        res_mask = np.ones(size,dtype=dtype)*mask_bg_value
        if w>0 and h>0:
            mask = masks[i]
            mask = cv2.resize(mask,(w,h))
            res_mask[y:y+h,x:x+w] = mask
        res_masks.append(res_mask)

    if len(res_masks)==0:
        return np.zeros([0,size[0],size[1]],dtype=dtype)
    return np.stack(res_masks,axis=0)

def draw_polygon(img,polygon,color=(255,255,255),is_line=True,isClosed=True):
    if is_line:
        return cv2.polylines(img, [polygon], color=color,isClosed=isClosed)
    else:
        return cv2.fillPoly(img,[polygon],color=color)
