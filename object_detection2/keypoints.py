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
keypoints:[X,N,2] [x,y]
'''
@btf.add_name_scope
def keypoints_flip_left_right(keypoints,swap_index=None):
    x,y = tf.unstack(keypoints,axis=-1)
    org_x = x
    cond = tf.logical_and(x>=0,y>=0)
    x = 1.0-x
    x = tf.where(cond,x,org_x)
    if swap_index is not None:
        swap_dict = {}
        for a,b in swap_index:
            swap_dict[a] = b
            swap_dict[b] = a
        X,N,_ = btf.combined_static_and_dynamic_shape(keypoints)
        indexs = []
        for i in range(N):
            if i in swap_dict:
                indexs.append(swap_dict[i])
            else:
                indexs.append(i)
        indexs = tf.convert_to_tensor(indexs,dtype=tf.int32)
        indexs = tf.reshape(indexs,[1,N])
        indexs = tf.tile(indexs,[X,1])
        x = tf.batch_gather(x,indexs)
        y = tf.batch_gather(y,indexs)

    return tf.stack([x,y],axis=-1)

'''
keypoints:[N,2] [x,y]
'''
@btf.add_name_scope
def keypoints_flip_up_down(keypoints):
    x,y = tf.unstack(keypoints,axis=-1)
    org_y = y
    cond = tf.logical_an(x>=0,y>=0)
    y = 1.0-y
    y = tf.where(cond,y,org_y)
    keypoints = tf.stack([x,y],axis=-1)
    return keypoints

'''
keypoints:[..,N,2] [x,y] absolute coordinate
'''
@btf.add_name_scope
def keypoits_rotate(keypoints,angle,width,height):
    r_angle = -angle * 3.1415926 / 180
    cos = tf.cos(r_angle)
    sin = tf.sin(r_angle)
    m = tf.stack([cos, -sin, sin, cos])
    m = tf.reshape(m,[2,2])
    old_shape = btf.combined_static_and_dynamic_shape(keypoints)
    keypoints = tf.reshape(keypoints,[-1,2])
    org_x,org_y = tf.unstack(keypoints,axis=-1)
    width = tf.to_float(width)
    height = tf.to_float(height)
    keypoints = keypoints-tf.convert_to_tensor([[(width-1)/2,(height-1)/2]])
    keypoints = tf.matmul(m,keypoints,transpose_b=True)
    keypoints = tf.transpose(keypoints,[1,0])+tf.convert_to_tensor([[(width-1)/2,(height-1)/2]])
    x,y = tf.unstack(keypoints,axis=-1)

    cond = tf.logical_and(org_x>=0,org_y>=0)
    x = tf.where(cond,x,org_x)
    y = tf.where(cond,y,org_y)

    cond = tf.logical_and(tf.logical_and(x>=0,x<width), tf.logical_and(y>=0,y<height))
    x = tf.where(cond,x,tf.ones_like(x)*-100)
    y = tf.where(cond,y,tf.ones_like(y)*-100)

    keypoints = tf.stack([x,y],axis=-1)
    keypoints = tf.reshape(keypoints,old_shape)
    return keypoints


@btf.add_name_scope
def batch_get_bboxes(keypoints, len):
    """
    get bboxes from keypoints
    Arguments:
        keypoints: (B,N,NUM_KEYPOINTS,2) (x,y)
        len: (B)
    Returns:
        bboxes: (B,N,4) (ymin,xmin,ymax,xmax)
    """
    bboxes = btf.static_or_dynamic_map_fn(
            lambda x: get_bboxes(x[0], x[1]),
            elems=[keypoints,len],
            dtype=tf.float32,
            back_prop=False)
    return bboxes

@btf.add_name_scope
def get_bboxes(keypoints,len):
    """
    get bboxes from keypoints
    Arguments:
        keypoints: (N,NUM_KEYPOINTS,2) (x,y)
        len: ()
    Returns:
        bboxes: (N,4) (ymin,xmin,ymax,xmax)
    """
    shape = btf.combined_static_and_dynamic_shape(keypoints)
    keypoints = keypoints[:len]
    bboxes = tf.map_fn(get_bbox,elems=keypoints,dtype=tf.float32,
                       back_prop=False)
    bboxes = tf.pad(bboxes,[[0,shape[0]-len],[0,0]])
    bboxes = tf.reshape(bboxes,[shape[0],4])

    return bboxes

@btf.add_name_scope
def get_bbox(keypoints):
    """
    get bboxes from keypoints
    Arguments:
        keypoints: (NUM_KEYPOINTS,2) (x,y)
    Returns:
        bboxes: (4) (ymin,xmin,ymax,xmax)
    """

    x,y = tf.unstack(keypoints,axis=-1)
    mask = tf.greater_equal(x,0)
    x = tf.boolean_mask(x,mask)
    y = tf.boolean_mask(y,mask)
    xmin = tf.reduce_min(x)
    xmax = tf.reduce_max(x)
    ymin = tf.reduce_min(y)
    ymax = tf.reduce_max(y)

    return tf.convert_to_tensor([ymin,xmin,ymax,xmax])
