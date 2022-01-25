import cv2
import numpy as np
import tensorflow as tf
import basic_tftools as btf
import math

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
    cond = tf.logical_and(x>=0,y>=0)
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

def npget_bbox(keypoints,threshold=0.02):
    '''

    Args:
        keypoints: [N,3] or [N,2] ,[x,y,visible]
        threshold:

    Returns:
        [xmin,ymin,xmax,ymax]

    '''
    if keypoints.shape[1]>=3:
        mask = keypoints[:,2]>threshold
        if np.any(mask):
            keypoints = keypoints[mask]
        else:
            return None
    xmin = np.min(keypoints[:,0])
    xmax = np.max(keypoints[:,0])
    ymin = np.min(keypoints[:,1])
    ymax = np.max(keypoints[:,1])
    return np.array([xmin,ymin,xmax,ymax],dtype=np.float32)

def npbatchget_bboxes(keypoints,threshold=0.02):
    if not isinstance(keypoints,np.ndarray):
        keypoints = np.array(keypoints)

    if len(keypoints.shape)==2:
        return npget_bbox(keypoints,threshold)

    bboxes = []
    for kps in keypoints:
        bboxes.append(npget_bbox(kps,threshold))
    return np.array(bboxes)

def keypoint_distance(kp0,kp1,use_score=True,score_threshold=0.1,max_dis=1e8):
    '''

    Args:
        kp0: [NP_NR,2/3] (x,y) or (x,y,score)
        kp1: [NP_NR,2/3] (x,y) or (x,y,score)
        use_score:
        score_threshold:
        max_dis:

    Returns:
    '''
    KP_NR = kp0.shape[0]
    NR_THRESHOLD = KP_NR/4
    def point_dis(p0,p1):
        dx = p0[0]-p1[0]
        dy = p0[1]-p1[1]
        return math.sqrt(dx*dx+dy*dy)

    if use_score:
        count_nr = 0
        sum_dis = 0.0
        for i in range(KP_NR):
            if kp0[i][2]>score_threshold and kp1[i][2]>score_threshold:
                count_nr += 1
                sum_dis += point_dis(kp0[i],kp1[i])
        if count_nr<=NR_THRESHOLD:
            return max_dis
        else:
            return sum_dis/count_nr
    else:
        sum_dis = 0.0
        for i in range(KP_NR):
            sum_dis += point_dis(kp0[i],kp1[i])

        return sum_dis/KP_NR

def keypoint_distancev2(kp0,kp1,bbox0,bbox1,use_score=True,score_threshold=0.1,max_dis=1e8):
    dis = keypoint_distance(kp0,kp1,use_score,score_threshold,max_dis)
    bboxes = np.stack([bbox0,bbox1],axis=0)
    hw = bboxes[...,2:]-bboxes[...,:2]
    size = np.maximum(1e-8,hw[...,0]*hw[...,1])
    size = np.sqrt(size)
    size = np.mean(size)
    return dis/size


def keypoints_distance(kps,use_score=True,score_threshold=0.1,max_dis=1e8):
    '''

    Args:
        kps: [N,KP_NR,3/2] (x,y) or (x,y,score)
        use_score:
        score_threshold:

    Returns:

    '''

    N = kps.shape[0]
    res = np.zeros([N],dtype=np.float32)

    for i in range(1,N):
        res[i] = keypoint_distance(kps[i-1],kps[i],use_score,score_threshold,max_dis)

    return res

def keypoints_distancev2(kps,bboxes,use_score=True,score_threshold=0.1,max_dis=1e8):
    '''

    Args:
        kps: [N,KP_NR,2/3]
        bboxes: [N,4] [ymin,xmin,ymax,xmax]
        use_score:
        score_threshold:
        max_dis:

    Returns:

    '''
    dis = keypoints_distance(kps,use_score,score_threshold,max_dis)

    hw = bboxes[...,2:]-bboxes[...,:2]
    size = np.maximum(1e-8,hw[...,0]*hw[...,1])
    size = np.sqrt(size)
    dis = dis/size
    return dis

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def rotate(angle,img,kps,bbox,scale=1.0):
    '''

    Args:
        angle: [-360,360]
        img: [RGB]
        kps: [N,2]/[N,3]
        bbox: [xmin,ymin,xmax,ymax]

    Returns:

    '''
    cx = (bbox[0]+bbox[2])/2
    cy = (bbox[1] + bbox[3]) / 2
    matrix = cv2.getRotationMatrix2D([cx,cy],angle,scale)
    img = cv2.warpAffine(img,matrix,dsize=(img.shape[1],img.shape[0]),
                         flags=cv2.INTER_LINEAR)
    num_joints = kps.shape[0]

    def is_good_point(p):
        return p[0]>=0 and p[1]>=0 and p[0]<img.shape[1] and p[1]<img.shape[0]

    for i in range(num_joints):
        if kps[i, 2] > 0.0:
            kps[i, 0:2] = affine_transform(kps[i, 0:2], matrix)
            if not is_good_point(kps[i]):
                kps[i,:] = 0
    points = [bbox[:2],[bbox[0],bbox[3]],bbox[2:],[bbox[2],bbox[1]]]
    res_points = []
    for p in points:
        p = affine_transform(np.array(p),matrix)
        res_points.append(p)
    res_points = np.array(res_points)
    x0y0 = np.maximum(np.min(res_points,axis=0),0)
    x1y1 = np.minimum(np.max(res_points,axis=0),[img.shape[1]-1,img.shape[0]-1])
    bbox = np.concatenate([x0y0,x1y1],axis=0)

    return img,kps,bbox

def fliplr_joints(joints, width, matched_parts,joints_vis):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        if joints_vis is not None:
            joints_vis[pair[0], :], joints_vis[pair[1], :] = \
                joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints,joints_vis

def flip(img,kps,joints_vis,matched_parts,bbox=None):
    '''

    Args:
        img:
        kps: [N,3] x,y,vis
        joints_vis: [N,3]
        matched_parts: [[id0_0,id0_1],...]
        bboxes: [xmin,ymin,xmax,ymax]

    Returns:

    '''
    img = img[:, ::-1, :]
    joints, joints_vis = fliplr_joints(
                    kps,img.shape[1], matched_parts,joints_vis)
    if bbox is not None:
        xmax = img.shape[1] - bbox[0] - 1
        bbox[0] = img.shape[1] - bbox[2] - 1
        bbox[2] = xmax

    return img,joints,joints_vis,bbox


def cut2size(kps,bbox,dst_size):
    '''
    bbox:[xmin,ymin,xmax,ymax]
    dst_size:[W,H]
    kps:[N,3]/[N,2]
    '''
    kps[...,:2] = kps[...,:2]-np.array([[bbox[0],bbox[1]]],dtype=np.float32)
    bbox_w = bbox[2]-bbox[0]
    bbox_h = bbox[3]-bbox[1]
    if bbox_w<1e-8 or bbox_h<1e-8:
        print(f"ERROR bbox")
    kps[...,:2] = kps[...,:2]*np.array([[dst_size[0]/bbox_w,dst_size[1]/bbox_h]],dtype=np.float32)
    return kps
