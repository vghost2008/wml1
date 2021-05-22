import tensorflow as tf
import basic_tftools as btf

'''
keypoints:[N,2] (x,y)
'''
def keypoints_rot90(keypoints,clockwise=True):
    x,y = tf.unstack(keypoints,axis=-1)
    if clockwise:
        x = 1.0-x
        keypoints = tf.stack([y,x],axis=1)
    else:
        y = 1.0-y
        keypoints = tf.stack([y,x],axis=1)
    return keypoints

'''
keypoints:[N,2],[x,y]
'''
@btf.add_name_scope
def keypoints_relative2absolute(keypoints,width,height):
    if not isinstance(width,tf.Tensor):
        width = tf.convert_to_tensor(width,dtype=keypoints.dtype)
    elif width.dtype != keypoints.dtype:
        width = tf.cast(width,keypoints.dtype)
    if not isinstance(height,tf.Tensor):
        height = tf.convert_to_tensor(height,dtype=keypoints.dtype)
    elif height.dtype != keypoints.dtype:
        height = tf.cast(height,keypoints.dtype)
    x = keypoints[...,0]*(width-1)
    y = keypoints[...,1]*(height-1)

    x_in_range = tf.logical_and(tf.greater_equal(x,0),tf.less_equal(x,width+2))
    x = tf.where(x_in_range,x,-tf.ones_like(x))
    y_in_range = tf.logical_and(tf.greater_equal(y,0),tf.less_equal(y,height+2))
    y = tf.where(y_in_range,y,-tf.ones_like(y))

    return tf.stack([x,y],axis=-1)

'''
keypoints:[N,2],[x,y]
'''
@btf.add_name_scope
def keypoints_absolute2relative(keypoints, width, height):
    if isinstance(height,tf.Tensor) and height.dtype != keypoints.dtype:
        height = tf.cast(height, keypoints.dtype)
    if isinstance(width,tf.Tensor) and width.dtype != keypoints.dtype:
        width = tf.cast(width, keypoints.dtype)
    x = keypoints[...,0] / (width-1)
    y = keypoints[...,1] / (height-1)

    x_in_range = tf.logical_and(tf.greater_equal(x,0),tf.less_equal(x,1.001))
    x = tf.where(x_in_range,x,-tf.ones_like(x))
    y_in_range = tf.logical_and(tf.greater_equal(y,0),tf.less_equal(y,1.001))
    y = tf.where(y_in_range,y,-tf.ones_like(y))

    return tf.stack([x,y], axis=-1)

'''
keypoints:[N,2] [x,y]
'''
@btf.add_name_scope
def keypoints_flip_left_right(keypoints):
    x,y = tf.unstack(keypoints,axis=-1)
    x = 1.0-x
    return tf.stack([x,y],axis=-1)

'''
keypoints:[N,2] [x,y]
'''
@btf.add_name_scope
def keypoints_flip_up_down(keypoints):
    x,y = tf.unstack(keypoints,axis=-1)
    y = 1.0-y
    keypoints = tf.stack([x,y],axis=-1)
    return keypoints