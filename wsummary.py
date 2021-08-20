#coding=utf-8
import tensorflow as tf
import cv2
import basic_tftools as btf
import image_visualization as imv
import semantic.visualization_utils as smv

isSingleValueTensor = btf.isSingleValueTensor

'''
images: [batch_size,H,W,C]
masks:shape=[batch_size,N,h,w]
boxes:shape=[batch_size,N,4]
size:[H,W]
mask_bg_value:mask background value
return:
shape=[batch_size,N,H,W]
'''
def detection_image_summary_with_croped_mask(images,
                           boxes,
                           *args,
                           instance_masks=None,
                           **kwargs):
    if instance_masks is not None:
        shape = btf.combined_static_and_dynamic_shape(images)
        assert len(instance_masks.get_shape()) == 4,"error mask dims"
        assert len(boxes.get_shape()) == 3,"error boxes dims"
        assert len(images.get_shape()) == 4,"error images dims"
        instance_masks = imv.batch_tf_get_fullsize_mask(boxes=boxes,
                                                        masks=instance_masks,
                                                        size=shape[1:3]
                                                        )
        if instance_masks.dtype != tf.uint8 and instance_masks.dtype != tf.bool:
            instance_masks = tf.cast(instance_masks > 0.5, tf.float32)
    return detection_image_summary(images,boxes,*args,instance_masks=instance_masks,**kwargs)

def detection_image_summary_by_logmask(images,
                            boxes,
                            classes=None,
                            scores=None,
                            category_index=None,
                            instance_masks=None,
                            keypoints=None,
                            logmask=None,
                            **kwargs):
    '''
    :param images:
    :param boxes:
    :param classes:
    :param scores:
    :param category_index:
    :param instance_masks:
    :param keypoints:
    :param logmask: to indict weather need log
    :param max_boxes_to_draw:
    :param min_score_thresh:
    :param name:
    :param max_outputs:
    :return:
    '''
    with tf.name_scope("detection_image_summary_by_logmask"):
        indices,lengths = btf.batch_mask_to_indices(logmask)
        #boxes = tf.Print(boxes, [tf.shape(boxes), tf.shape(images), lengths, logmask, indices], name="XXXXX",
                         #summarize=1000)
        if boxes is not None:
            boxes = btf.batch_gather(boxes,indices)
        if classes is not None:
            classes = btf.batch_gather(classes,indices)
        if scores is not None:
            scores = btf.batch_gather(scores,indices)
        if keypoints is not None:
            keypoints = btf.batch_gather(keypoints,indices)
        if instance_masks is not None:
            instance_masks = btf.batch_gather(instance_masks,indices)
        return detection_image_summary(images,boxes,classes,scores,category_index,instance_masks,keypoints,
                                       lengths,**kwargs)


def detection_image_summary(images,
                           boxes,
                           classes=None,
                           scores=None,
                           category_index=None,
                           instance_masks=None,
                           keypoints=None,
                           lengths=None,
                           max_boxes_to_draw=20,
                           min_score_thresh=0.2,name="detection_image_summary",max_outputs=3):
  """Draws bounding boxes, masks, and keypoints on batch of image tensors.

  Args:
    images: A 4D uint8 image tensor of shape [N, H, W, C].
    boxes: [N, max_detections, 4] float32 tensor of detection boxes.
    classes: [N, max_detections] int tensor of detection classes. Note that
      classes are 1-indexed.
    scores: [N, max_detections] float32 tensor of detection scores.
    category_index: a dict that maps integer ids to category dicts. e.g.
      {1:  'dog', 2:'cat', ...}
    instance_masks: A 4D uint8 tensor of shape [N, max_detection, H, W] with
      instance masks.
    keypoints: A 4D float32 tensor of shape [N, max_detection, num_keypoints, 2]
      with keypoints.
    max_boxes_to_draw: Maximum number of boxes to draw on an image. Default 20.
    min_score_thresh: Minimum score threshold for visualization. Default 0.2.

  Returns:
    4D image tensor of type uint8, with boxes drawn on top.
  """
  assert len(boxes.get_shape()) == 3, f"error boxes dims {len(boxes.get_shape())}"
  assert len(images.get_shape()) == 4, f"error images dims {len(images.get_shape())}"
  images = images[:max_outputs]
  boxes = boxes[:max_outputs]
  if classes is not None:
      classes = classes[:max_outputs]
      assert len(classes.get_shape()) == 2, f"error classes dims {len(classes.get_shape())}"
  if scores is not None:
      scores = scores[:max_outputs]
      assert len(scores.get_shape()) == 2, f"error scores dims {len(scores.get_shape())}"
  if instance_masks is not None:
      instance_masks = instance_masks[:max_outputs]
      assert len(instance_masks.get_shape()) == 4, f"error instance mask dims {len(instance_masks.get_shape())}"
  if keypoints is not None:
      keypoints = keypoints[:max_outputs]
  if lengths is not None:
      lengths = lengths[:max_outputs]
      assert len(lengths.get_shape()) == 1, f"error length dims {len(lengths.get_shape())}"

  with tf.device(":/cpu:0"):
    if images.get_shape().as_list()[0] is None:
        nr = tf.reduce_prod(tf.shape(boxes))
        B,H,W,C = btf.combined_static_and_dynamic_shape(images)
        images = tf.cond(tf.greater(B,0),lambda:images,lambda:tf.ones([1,H,W,C],dtype=images.dtype))
        boxes = tf.cond(tf.greater(nr,0),lambda:boxes,lambda:tf.ones([1,1,4],dtype=boxes.dtype))
        if classes is not None:
            classes = tf.cond(tf.greater(nr, 0), lambda: classes, lambda: tf.ones([1, 1], dtype=classes.dtype))
        if scores is not None:
            scores = tf.cond(tf.greater(nr, 0), lambda: scores, lambda: tf.ones([1, 1], dtype=scores.dtype))
        if instance_masks is not None:
            B, N,H, W = btf.combined_static_and_dynamic_shape(instance_masks)
            instance_masks = tf.cond(tf.greater(nr, 0), lambda: instance_masks, lambda: tf.zeros([1, 1,H,W],
                                                                                                 dtype=instance_masks.dtype))

    images = imv.draw_detection_image_summary(images,
                                         boxes,
                                         classes,
                                         scores,
                                         category_index,
                                         instance_masks,
                                         keypoints,
                                         lengths=lengths,
                                         max_boxes_to_draw=max_boxes_to_draw,
                                         min_score_thresh=min_score_thresh)

  tf.summary.image(name,images,max_outputs=max_outputs)


def variable_summaries(var,name):
    if var.dtype != tf.float32:
        var = tf.cast(var, tf.float32)
    mean = tf.reduce_mean(var)
    tf.summary.scalar(name+'/max', tf.reduce_max(var))
    tf.summary.scalar(name+'/min', tf.reduce_min(var))
    tf.summary.scalar(name+'/mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(name+'/stddev', stddev)
    tf.summary.histogram(name+'/hisogram', var)

def variable_summaries_v2(var, name,family=None):
    if var is None or name is None:
        return
    if var.dtype != tf.float32:
        var = tf.cast(var, tf.float32)
    if isSingleValueTensor(var):
        var = tf.reshape(var,())
        tf.summary.scalar(name, var,family=family)
        return
    mean = tf.reduce_mean(var)
    tf.summary.scalar(name+'/max', tf.reduce_max(var),family=family)
    tf.summary.scalar(name+'/min', tf.reduce_min(var),family=family)
    tf.summary.scalar(name+'/mean', mean,family=family)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(name+'/stddev', stddev,family=family)
    tf.summary.histogram(name+'/hisogram', var,family=family)

def histogram(var,name,family=None):
    if not isSingleValueTensor(var):
        tf.summary.histogram(name, var,family=family)

def histogram_or_scalar(var,name,family=None):
    if not isSingleValueTensor(var):
        tf.summary.histogram(name, var,family=family)
    else:
        var = tf.reshape(var,())
        tf.summary.scalar(name, var,family=family)

def image_summaries(var,name,max_outputs=3):
    if var.get_shape().ndims==3:
        var = tf.expand_dims(var,dim=0)
    tf.summary.image(name,var,max_outputs=max_outputs)

def _draw_text_on_image(img,text,font_scale=1.2,color=(0.,255.,0.),pos=None):
    if isinstance(text,bytes):
        text = str(text,encoding="utf-8")
    if not isinstance(text,str):
        text = str(text)
    thickness = 2
    size = cv2.getTextSize(text,fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=font_scale,thickness=thickness)
    if pos is None:
        pos = (0,(img.shape[0]+size[0][1])//2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale, color=color, thickness=thickness)
    return img

def image_summaries_with_label(img,label,name,max_outputs=3,scale=True):
    shape = img.get_shape().as_list()
    if shape[0] is not None:
        img = img[:max_outputs,:,:,:]
        label = label[:max_outputs]
    if scale:
        img = (img+1.0)*127.5

    def draw_func(var,l):
        return tf.py_func(_draw_text_on_image,[var,l],var.dtype)

    img = tf.cond(tf.greater(tf.shape(label)[0],0),lambda:btf.static_or_dynamic_map_fn(lambda x:draw_func(x[0],x[1]),elems=[img,label],dtype=img.dtype),
                  lambda:img)
    tf.summary.image(name,img,max_outputs=max_outputs)

def row_image_summaries(imgs,name="image_contrast",max_outputs=3,margin=10,resize=False,is_hsv=False):
    with tf.name_scope(name):
        is_channel_equal = True
        shape = btf.combined_static_and_dynamic_shape(imgs[0])
        log_image = tf.identity(imgs[0][:max_outputs])
        channel = log_image.get_shape().as_list()[-1]
        for i in range(1,len(imgs)):
            if imgs[i].get_shape().as_list()[-1] != channel:
                is_channel_equal = False
                break
        if not is_channel_equal:
            if log_image.get_shape().as_list()[-1] == 1:
                log_image = tf.tile(log_image,[1,1,1,3])
        for i in range(1,len(imgs)):
            log_image = tf.pad(log_image, paddings=[[0, 0], [0, 0], [0, margin], [0, 0]])
            img = imgs[i][:max_outputs]
            if resize:
                img = tf.image.resize_nearest_neighbor(img,size=shape[1:3])
            if not is_channel_equal:
                if img.get_shape().as_list()[-1] == 1:
                    img = tf.tile(img,[1,1,1,3])
            log_image = tf.concat([log_image,img ], axis=2)
        if is_hsv:
            log_image = tf.image.hsv_to_rgb(log_image)*2.0-1.0
        image_summaries(log_image, "image_contrast")

def positive_box_on_images_summary(image,boxes,pmasks,name="positive_box_on_images_summary"):
    image = imv.draw_positive_box_on_images(image,boxes,pmasks)
    image_summaries(image,name)

@btf.add_name_scope
def feature_map_summary(feature_map,name="feature_map",max_outputs=None,min_max=None):
    data = feature_map[0]
    if data.dtype != tf.float32:
        data = tf.cast(data,tf.float32)
    if max_outputs is None:
        max_outputs = feature_map.get_shape().as_list()[-1]
    data = tf.transpose(data,[2,0,1])
    data = tf.expand_dims(data,axis=-1)
    if min_max is None:
        min = tf.reduce_min(data)
        max = tf.reduce_max(data)
    else:
        min,max = min_max
    data = (data-min)/(max-min+1e-8)
    tf.summary.image(name,data,max_outputs=max_outputs)

def summary_image_with_box(image,
                                bboxes,
                                classes,
                                scores=None,
                                category_index=None,
                                max_boxes_to_draw=100,
                                name='summary_image_with_box',scale=True):
    with tf.name_scope(name):
        if scale:
            if (image.dtype==tf.float32) or (image.dtype==tf.float16) or (image.dtype==tf.float64):
                #floatpoint data value range is [-1,1]
                image = (image+1.0)*127.5
                image = tf.clip_by_value(image,clip_value_min=0.,clip_value_max=255.)
                image = tf.cast(image,dtype=tf.uint8)
        if image.get_shape().ndims == 3:
            image = tf.expand_dims(image,axis=0)
        if image.dtype is not tf.uint8:
            image = tf.cast(image,tf.uint8)
        if bboxes.get_shape().ndims == 2:
            bboxes = tf.expand_dims(bboxes, 0)
        if scores is None:
            scores = tf.ones(shape=tf.shape(bboxes)[:2],dtype=tf.float32)
        elif scores.get_shape().ndims ==1:
            scores = tf.expand_dims(scores,axis=0)
        if category_index is None:
            category_index = {}
            for i in range(100):
                category_index[i] = {"name":f"{i}"}
        if classes.get_shape().ndims == 1:
            classes = tf.expand_dims(classes,axis=0)
        image_with_box = smv.draw_bounding_boxes_on_image_tensors(image,bboxes,classes,scores,
                                                              category_index=category_index,
                                                              max_boxes_to_draw=max_boxes_to_draw)
        tf.summary.image(name, image_with_box)

def summary_graph(img,points,adj_mt,name="summary_graph",is_relative_coordinate=True):
    img = imv.draw_graph(img,points,adj_mt,is_relative_coordinate)
    tf.summary.image(name,tf.expand_dims(img,axis=0))

def summary_graph_by_bboxes(img,bboxes,adj_mt,name="summary_graph",is_relative_coordinate=True):
    ymin,xmin,ymax,xmax = tf.unstack(bboxes,axis=-1)
    cx = (xmin+xmax)/2
    cy = (ymin+ymax)/2
    points = tf.stack([cx,cy],axis=-1)
    summary_graph(img,points,adj_mt,name,is_relative_coordinate)

def semantic_summary_img(img,masks,colors=None,name="semantic"):
    ''''
    image: [height, width, 3] value range[0, 255]
    mask: [height, width,N] value range[0, 1]
    '''
    with tf.device("/cpu:0"):
        masks = tf.transpose(masks,[2,0,1])
        if img.dtype != tf.uint8:
            min = tf.reduce_min(img)
            max = tf.reduce_max(img)
            img = (img-min)*255/(max-min+1e-8)
            img = tf.cast(img,tf.uint8)
        if colors is None:
            colors = smv.STANDARD_COLORS
        colors = tf.convert_to_tensor(colors)
        img = smv.tf_draw_masks_on_image(img,masks,color=colors)
    return img

def semantic_summary(img,masks,colors=None,name="semantic"):
    ''''
    image: [height, width, 3] value range[0, 255]
    mask: [height, width,N] value range[0, 1]
    '''
    img = semantic_summary_img(img,masks,colors,name)
    tf.summary.image(name,tf.expand_dims(img,axis=0))

def batch_semantic_summary(img,masks,max_outputs=3,colors=None,name="semantic"):
    ''''
    image: [B,height, width, 3] value range[0, 255]
    mask: [B, height, width,N] value range[0, 1]
    '''
    with tf.device("/cpu:0"):
        img = img[:max_outputs]
        masks = masks[:max_outputs]
        img = tf.map_fn(lambda x:semantic_summary_img(x[0],x[1],colors),elems=(img,masks),dtype=tf.uint8,
                        back_prop=False)
        tf.summary.image(name,img)

def keypoints_image_summary(images,
                            keypoints=None,
                            lengths=None,
                            max_instance_to_draw=20,
                            keypoints_pair=None,
                            name="keypoints_image_summary",max_outputs=3):
    """Draws bounding keypoints on batch of image tensors.

    Args:
      images: A 4D uint8 image tensor of shape [N, H, W, C].
      keypoints: A 4D float32 tensor of shape [N, max_detection, num_keypoints, 2] (x,y)
        with keypoints.
      max_instance_to_draw: Maximum number of instance to draw on an image. Default 20.

    Returns:
      4D image tensor of type uint8, with boxes drawn on top.
    """
    assert len(keypoints.get_shape()) == 4, f"error keypoints dims {len(keypoints.get_shape())}"
    assert len(images.get_shape()) == 4, f"error images dims {len(images.get_shape())}"
    images = images[:max_outputs]
    keypoints = keypoints[:max_outputs]
    if lengths is not None:
        lengths = lengths[:max_outputs]
        assert len(lengths.get_shape()) == 1, f"error length dims {len(lengths.get_shape())}"

    with tf.device(":/cpu:0"):
        if images.get_shape().as_list()[0] is None:
            nr = tf.reduce_prod(tf.shape(keypoints))
            B,H,W,C = btf.combined_static_and_dynamic_shape(images)
            B,MD,num_keypoints,_ = btf.combined_static_and_dynamic_shape(images)
            images = tf.cond(tf.greater(B,0),lambda:images,lambda:tf.ones([1,H,W,C],dtype=images.dtype))
            boxes = tf.cond(tf.greater(nr,0),lambda:boxes,lambda:tf.ones([1,1,4],dtype=boxes.dtype))
            keypoints = tf.cond(tf.greater(nr, 0), lambda: keypoints, lambda: tf.zeros([B, 1,num_keypoints,2], dtype=keypoints.dtype))
        images = imv.draw_keypoints_image_summary(images,
                                                  keypoints,
                                                  keypoints_pair=keypoints_pair,
                                                  lengths=lengths,
                                                  max_instance_to_draw=max_instance_to_draw)

    tf.summary.image(name,images,max_outputs=max_outputs)