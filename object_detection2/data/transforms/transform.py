#coding=utf-8
from itertools import count

import cv2
import tensorflow as tf
import img_utils as wmli
from collections import Iterable
import object_detection2.bboxes as wml_bboxes
import time
from functools import partial
import basic_tftools as btf
from object_detection2.standard_names import *
import numpy as np
from .autoaugment import *
from thirdparty.aug.autoaugment import distort_image_with_autoaugment
import object_detection2.bboxes as odb
import object_detection2.od_toolkit as odt
from basic_tftools import channel
import copy
import random
import wnnlayer as wnnl
import tfop
import object_detection2.keypoints as kp
from .transform_toolkit import motion_blur

'''
所有的变换都只针对一张图, 部分可以兼容同时处理一个batch
'''
class WTransform(object):
    GLOBAL_STATUS = 0
    ABSOLUTE_COORDINATE = 1
    HWN_MASK = 2
    
    @staticmethod
    def test_statu(statu):
        return (WTransform.GLOBAL_STATUS&statu)>0
    
    @staticmethod
    def test_unstatu(statu):
        return (WTransform.GLOBAL_STATUS&statu)==0

    @staticmethod
    def set_statu(statu):
        WTransform.GLOBAL_STATUS = WTransform.GLOBAL_STATUS|statu
        
    @staticmethod
    def unset_statu(statu):
        WTransform.GLOBAL_STATUS = WTransform.GLOBAL_STATUS^statu

    def is_image(self,key):
        return ('image' in key) or ('img' in key)

    def is_mask(self,key):
        return ('mask' in key)

    def is_image_or_mask(self,key):
        return ('image' in key) or ('img' in key) or ('mask' in key)

    def is_bbox(self,key):
        return ('box' in key)

    def is_keypoints(self,key):
        return ('keypoint' in key)

    def is_label(self,key):
        return ('label' in key) or ('class' in key)

    '''
    func: func(x:Tensor)->Tensor
    '''
    def apply_to_images(self,func,data_item,**kwargs):
        return self.apply_with_filter(func,data_item,filter=self.is_image,**kwargs)

    def apply_to_masks(self,func,data_item,**kwargs):
        return self.apply_with_filter(func,data_item,filter=self.is_mask,**kwargs)

    def apply_to_images_and_masks(self,func,data_item,**kwargs):
        return self.apply_with_filter(func,data_item,
                                      filter=self.is_image_or_mask,**kwargs)

    def apply_to_bbox(self,func,data_item,**kwargs):
        return self.apply_with_filter(func,data_item,filter=self.is_bbox,**kwargs)

    def apply_to_keypoints(self,func,data_item,**kwargs):
        return self.apply_with_filter(func,data_item,filter=self.is_keypoints,**kwargs)

    def apply_to_label(self,func,data_item,**kwargs):
        return self.apply_with_filter(func,data_item,filter=self.is_label,**kwargs)

    def apply_with_filter(self,func,data_item,filter,runtime_filter=None):
        res = {}
        if runtime_filter is None:
            for k,v in data_item.items():
                if filter(k):
                    res[k] = func(v)
                else:
                    res[k] = v
        else:
            for k,v in data_item.items():
                if filter(k):
                    res[k] = tf.cond(runtime_filter,lambda:func(v),lambda:v)
                else:
                    res[k] = v
        return res

    def __repr__(self):
        return type(self).__name__

    @staticmethod
    def select(pred,true_v,false_v):
        return tf.cond(pred,lambda:true_v,lambda:false_v)

    @staticmethod
    def cond_set(dict_data,key,pred,v):
        if pred is None:
            dict_data[key] = v
        else:
            dict_data[key] = tf.cond(pred,lambda:v,lambda:dict_data[key])

    @staticmethod
    def cond_fn_set(dict_data,key,pred,fn):
        dict_data[key] = tf.cond(pred,fn,lambda:dict_data[key])

    @staticmethod
    def probability_set(dict_data,key,prob,v):
        pred = tf.less_equal(tf.random_uniform(shape=()),prob)
        dict_data[key] = tf.cond(pred,lambda:v,lambda:dict_data[key])
        
    @staticmethod
    def probability_fn_set(dict_data,key,prob,fn,fn2=None):
        pred = tf.less_equal(tf.random_uniform(shape=()),prob)
        if fn2 is None:
            dict_data[key] = tf.cond(pred,fn,lambda:dict_data[key])
        else:
            dict_data[key] = tf.cond(pred,fn,fn2)

    @staticmethod
    def pad(dict_data,key,paddings):
        data = dict_data[key]
        if len(paddings)<len(data.get_shape()):
            paddings = paddings+[[0,0]]*(len(data.get_shape())-len(paddings))
        dict_data[key] = tf.pad(data,paddings=paddings)
'''
img:[H,W,C]
'''
def random_blur(img,size=(5,5),sigmaX=0,sigmaY=0,prob=0.5):
    return tf.cond(tf.greater(tf.random_uniform(shape=[]), prob),
                   lambda: (img),
                   lambda: wmli.blur(img,size,sigmaX,sigmaY))

'''
img:[H,W,C] or [N,H,W,C]
limit: 参考tfop.get_image_resize_size
align: 参考tfop.get_image_resize_size
'''
def resize_img(img,limit,align,resize_method=tf.image.ResizeMethod.BILINEAR):
    with tf.name_scope("resize_img"):
        new_size = tfop.get_image_resize_size(size=tf.shape(img)[0:2], limit=limit, align=align)
        return tf.image.resize_images(img, new_size,method=resize_method)

def distort_color(image, color_ordering=2, fast_mode=False,
            b_max_delta=0.1,
            c_lower = 0.8,
            c_upper = 1.2,
            s_lower = 0.5,
            s_upper = 1.5,
            h_max_delta = 0.1,
            scope=None,seed=None):
    with tf.name_scope(scope, 'distort_color', [image]):
        ori_dtype = image.dtype
        if ori_dtype != tf.uint8:
            print("WARNING: distort color should performace on tf.uint8.")
        image = tf.image.convert_image_dtype(image,tf.float32)
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:

            if color_ordering == 0:
                image = tf.image.random_brightness(image, b_max_delta)
                image = tf.image.random_contrast(image, lower=c_lower, upper=c_upper)
            elif color_ordering == 1:
                image = tf.image.random_contrast(image, lower=c_lower, upper=c_upper)
                image = tf.image.random_brightness(image, b_max_delta)
            elif color_ordering == 2:
                image = tf.image.random_brightness(image, b_max_delta)
                image = tf.image.random_contrast(image, lower=c_lower, upper=c_upper)
                image = tf.image.random_saturation(image, lower=s_lower, upper=s_upper)
            elif color_ordering == 3:
                image = tf.image.random_saturation(image, lower=s_lower, upper=s_upper)
                image = tf.image.random_brightness(image, b_max_delta)
                image = tf.image.random_contrast(image, lower=c_lower, upper=c_upper)
            elif color_ordering == 4:
                image = tf.image.random_saturation(image, lower=s_lower, upper=s_upper)
                image = tf.image.random_brightness(image, b_max_delta)
                image = tf.image.random_contrast(image, lower=c_lower, upper=c_upper)
                image = tf.image.random_hue(image, max_delta=h_max_delta)
            elif color_ordering == 5:
                image = tf.image.random_hue(image, max_delta=h_max_delta)
                image = tf.image.random_saturation(image, lower=s_lower, upper=s_upper)
                image = tf.image.random_brightness(image, b_max_delta)
                image = tf.image.random_contrast(image, lower=c_lower, upper=c_upper)
            elif color_ordering == 6:
                image = tf.image.random_hue(image, max_delta=h_max_delta,seed=seed)
                image = tf.image.random_brightness(image, b_max_delta,seed=seed)
                image = tf.image.random_contrast(image, lower=c_lower, upper=c_upper,seed=seed)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        if ori_dtype.is_integer:
            scale = ori_dtype.max + 0.5
            image = image*scale
        return image

def random_crop(image,
                labels,
                bboxes,
                crop_size=(224,224),
                filter_threshold=0.3,
                scope=None):
    '''
    data argument for object detection
    :param image: [height,width,channels], input image
    :param labels: [num_boxes]
    :param bboxes: [num_boxes,4] (ymin,xmin,ymax,xmax), relative coordinates
    :param scope:
    :return:
    croped_image:[h,w,C]
    labels:[n]
    bboxes[n,4]
    mask:[num_boxes]
    bbox_begin:[3]
    bbox_size:[3]
    '''
    with tf.name_scope(scope, 'random_crop', [image, bboxes]):
        '''
        在满足一定要求的前提下对图片进行随机的裁剪
        '''
        bbox = odb.get_random_crop_bboxes(image,crop_size)
        bbox_begin = tf.concat([bbox[:2],tf.convert_to_tensor([0],tf.int32)],axis=0)
        bbox_size = tf.concat([bbox[2:]-bbox[:2],tf.convert_to_tensor([-1],tf.int32)],axis=0)
        img_shape = btf.combined_static_and_dynamic_shape(image)
        fbbox = tf.cast(bbox,tf.float32)
        distort_bbox = odb.tfabsolutely_boxes_to_relative_boxes(fbbox,width=img_shape[1],height=img_shape[0])
        '''
        distort_bbox的shape一定为[4],表示需要裁剪的裁剪框，与bbox_begin,bbox_size表达的内容重复
        '''
        image_channel = image.get_shape().as_list()[-1]
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        cropped_image.set_shape([None, None, image_channel])
        #cropped_image.set_shape([crop_size[0], crop_size[1], image_channel])

        '''
        将在原图中的bboxes转换为在distort_bbox定义的子图中的bboxes
        保留了distort_bbox界外的部分
        '''
        bigger_bboxes = wml_bboxes.bboxes_resize(distort_bbox, bboxes)
        '''
        仅保留交叉面积大于threshold的bbox
        '''
        labels, bboxes,mask,ignore_mask = wml_bboxes.bboxes_filter_overlap(labels, bigger_bboxes,
                                                                           threshold=filter_threshold,
                                                                           assign_negative=False,
                                                                           return_ignore_mask=True)
        ignore_bboxes = tf.boolean_mask(bigger_bboxes,ignore_mask)
        ignore_bboxes = odb.tf_correct_yxminmax_boxes(ignore_bboxes)
        H,W,C = btf.combined_static_and_dynamic_shape(cropped_image)
        ignore_bboxes = odb.tfrelative_boxes_to_absolutely_boxes(ignore_bboxes,width=W,height=H)
        #cropped_image = tfop.fill_bboxes(image=cropped_image,bboxes=ignore_bboxes,v=127.5,include_last=True)
        return cropped_image, labels, bboxes,bbox_begin,bbox_size,mask

def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered=0.3,
                                aspect_ratio_range=(0.9, 1.1),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                filter_threshold=0.3,
                                use_image_if_no_bounding_boxes=True,
                                scope=None):
    '''
    data argument for object detection
    :param image: [height,width,channels], input image
    :param labels: [num_boxes]
    :param bboxes: [num_boxes,4] (ymin,xmin,ymax,xmax), relative coordinates
    :param min_object_covered: the croped object bbox and the orignal object bbox's IOU must greater than
    min_object_covered to be keep in the result, else thre object will be removed.
    :param aspect_ratio_range: random crop aspect ratio range
    :param area_range:
    :param max_attempts:
    :param scope:
    :return:
    croped_image:[h,w,C]
    labels:[n]
    bboxes[n,4]
    mask:[num_boxes]
    bbox_begin:[3]
    bbox_size:[3]
    '''
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes]):
        '''
        在满足一定要求的前提下对图片进行随机的裁剪
        '''
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=tf.expand_dims(bboxes, 0),
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes,
                seed=int(time.time()),
                seed2=int(10*time.time()))
        '''
        distort_bbox的shape一定为[1, 1, 4],表示需要裁剪的裁剪框，与bbox_begin,bbox_size表达的内容重复
        '''
        distort_bbox = distort_bbox[0, 0]
        image_channel = image.get_shape().as_list()[-1]
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        cropped_image.set_shape([None, None, image_channel])

        '''
        将在原图中的bboxes转换为在distort_bbox定义的子图中的bboxes
        保留了distort_bbox界外的部分
        '''
        bigger_bboxes = wml_bboxes.bboxes_resize(distort_bbox, bboxes)
        '''
        仅保留交叉面积大于threshold的bbox
        '''
        labels, bboxes,mask,ignore_mask = wml_bboxes.bboxes_filter_overlap(labels, bigger_bboxes,
                                                               threshold=filter_threshold,
                                                               assign_negative=False,
                                                               return_ignore_mask=True)
        ignore_bboxes = tf.boolean_mask(bigger_bboxes,ignore_mask)
        ignore_bboxes = odb.tf_correct_yxminmax_boxes(ignore_bboxes)
        H,W,C = btf.combined_static_and_dynamic_shape(cropped_image)
        ignore_bboxes = odb.tfrelative_boxes_to_absolutely_boxes(ignore_bboxes,width=W,height=H)
        cropped_image = tfop.fill_bboxes(image=cropped_image,bboxes=ignore_bboxes,v=127.5,include_last=True)
        return cropped_image, labels, bboxes,bbox_begin,bbox_size,mask

def random_rot90(image,bboxes,clockwise=True):
    return tf.cond(tf.greater(tf.random_uniform(shape=[]), 0.5),
                   lambda: (image, bboxes),
                   lambda: rot90(image,bboxes,clockwise))

def rot90(image,bboxes,clockwise=True):
    if clockwise:
        k = 1
    else:
        k = 3
    if isinstance(image,list):
        image = [tf.cond(tf.reduce_min(tf.shape(img))>0,lambda:tf.image.rot90(img,k),lambda:img) for img in image]
    else:
        image = tf.image.rot90(image,k)
    return image,wml_bboxes.bboxes_rot90(bboxes,clockwise)

def random_flip_left_right(image,bboxes):
    return tf.cond(tf.greater(tf.random_uniform(shape=[]), 0.5),
            lambda: (image, bboxes),
            lambda: wml_bboxes.flip_left_right(image,bboxes))
'''
image: A 3-D tensor of shape [height, width, channels].
bboxes:[X,4] X个bboxes,使用相对坐标，[ymin,xmin,ymax,xmax]
'''
def flip_up_down(image,bboxes):
    if isinstance(image,list):
        image = [tf.cond(tf.reduce_min(tf.shape(img))>0,lambda:tf.image.flip_up_down(img),lambda:img) for img in image]
    else:
        image = tf.image.flip_up_down(image)
    return image,wml_bboxes.bboxes_flip_up_down(bboxes)

'''
Boxes: relative coordinate
mask:H,W,N format
'''
class RandomFlipLeftRight(WTransform):
    def __init__(self,cfg=None):
        self.cfg = cfg

    def __call__(self, data_item):
        if not self.test_unstatu(WTransform.ABSOLUTE_COORDINATE):
            print(f"WARNING: {self} need relative coordinate.")
        is_flip = tf.greater(tf.random_uniform(shape=[]),0.5)
        func = tf.image.flip_left_right
        data_item = self.apply_to_images_and_masks(func,data_item,runtime_filter=is_flip)
        if GT_BOXES in data_item:
            func2 = wml_bboxes.bboxes_flip_left_right
            data_item = self.apply_to_bbox(func2,data_item,runtime_filter=is_flip)
        if GT_KEYPOINTS:
            if self.cfg is not None:
                swap_index = self.cfg.MODEL.KEYPOINTS.POINTS_LEFT_RIGHT_GROUP
                X,N,C = data_item[GT_KEYPOINTS].shape.as_list()
                if N is None:
                    N = self.cfg.MODEL.KEYPOINTS.NUM_KEYPOINTS
                    data_item[GT_KEYPOINTS].set_shape([X,N,C])
            else:
                swap_index = None
            func3 = partial(kp.keypoints_flip_left_right,swap_index=swap_index)
            data_item = self.apply_to_keypoints(func3, data_item, runtime_filter=is_flip)

        return data_item
    
    def __repr__(self):
        return f"{RandomFlipLeftRight.__name__}"

'''
Boxes: relative coordinate
mask:H,W,N format
'''
class RandomFlipUpDown(WTransform):
    def __init__(self):
        pass
    def __call__(self, data_item):
        if not self.test_unstatu(WTransform.ABSOLUTE_COORDINATE):
            print(f"WARNING: {self} need relative coordinate.")
        is_flip = tf.greater(tf.random_uniform(shape=[]),0.5)
        func = tf.image.flip_up_down
        data_item = self.apply_to_images_and_masks(func,data_item,runtime_filter=is_flip)
        if GT_BOXES:
            func2 = wml_bboxes.bboxes_flip_up_down
            data_item = self.apply_to_bbox(func2,data_item,runtime_filter=is_flip)
        if GT_KEYPOINTS:
            func3 = kp.keypoints_flip_up_down
            data_item = self.apply_to_keypoints(func3, data_item, runtime_filter=is_flip)
        return data_item

    def __repr__(self):
        return f"{RandomFlipUpDown.__name__}"

class AutoAugment(WTransform):
    def __init__(self,augmentation_name="v0"):
        self.augmentation_name = augmentation_name
    def __call__(self, data_item:dict):
        if GT_MASKS in data_item:
            data_item.pop(GT_MASKS)
        image,bbox = distort_image_with_autoaugment(data_item[IMAGE],data_item[GT_BOXES],self.augmentation_name)
        data_item[IMAGE] = image
        data_item[GT_BOXES] = bbox
        if GT_KEYPOINTS in data_item:
            print(f"WARNING: keypoints don't support transform {self}.")

        return data_item
'''
Boxes: relative coordinate
mask:H,W,N format
'''
class RandomRotate(WTransform):
    def __init__(self,clockwise=True,probability=0.5):
        self.clockwise = clockwise
        if clockwise:
            self.k = 1
        else:
            self.k = 3
        self.probability = probability

    def __call__(self, data_item):
        if not self.test_unstatu(WTransform.ABSOLUTE_COORDINATE):
            print(f"WARNING: {self} need relative coordinate.")
        is_rotate = tf.less_equal(tf.random_uniform(shape=[]),self.probability)
        func = partial(tf.image.rot90,k=self.k)
        data_item = self.apply_to_images_and_masks(func,data_item,runtime_filter=is_rotate)
        if GT_BOXES in data_item:
            func2 = partial(wml_bboxes.bboxes_rot90,clockwise=self.clockwise)
            data_item = self.apply_to_bbox(func2,data_item,runtime_filter=is_rotate)
        if GT_KEYPOINTS in data_item:
            func3 = partial(kp.keypoints_rot90, clockwise=self.clockwise)
            data_item = self.apply_to_keypoints(func3, data_item, runtime_filter=is_rotate)
        return data_item
    
    def __repr__(self):
        return f"{RandomRotate.__name__}"

class RemoveZeroAreaBBox(WTransform):
    def __init__(self,mini_area=1e-4):
        self.mini_area = mini_area

    def __call__(self, data_item):
        if not self.test_statu(WTransform.ABSOLUTE_COORDINATE):
            print(f"INFO: {self} better use absolute coordinate.")
        if GT_MASKS in data_item:
            if not self.test_unstatu(WTransform.HWN_MASK):
                print(f"WARNING: {self} need NHW format mask.")

        area = odb.box_area(data_item[GT_BOXES])
        keep = tf.greater_equal(area,self.mini_area)
        data_item[GT_BOXES] = tf.boolean_mask(data_item[GT_BOXES],keep)
        data_item[GT_LABELS] = tf.boolean_mask(data_item[GT_LABELS],keep)
        if GT_MASKS in data_item:
            data_item[GT_MASKS] = tf.boolean_mask(data_item[GT_MASKS], keep)
        if GT_KEYPOINTS in data_item:
            data_item[GT_KEYPOINTS] = tf.boolean_mask(data_item[GT_KEYPOINTS], keep)

        return data_item
    
    def __repr__(self):
        return f"{RemoveZeroAreaBBox.__name__}"


'''
Mask: [Nr,H,W]
bboxes: absolute coordinate
'''
class RandomRotateAnyAngle(WTransform):
    def __init__(self,max_angle=60,rotate_probability=0.5,enable=True,use_mask=True,rotate_bboxes_type=0):
        self.max_angle = max_angle
        self.rotate_probability = rotate_probability
        self.enable = enable
        self.use_mask = use_mask
        self.rotate_bboxes_type = rotate_bboxes_type

    def __call__(self, data_item):
        if not self.test_statu(WTransform.ABSOLUTE_COORDINATE):
            print("------------------------------------------------")
            print(f"ERROR: {self} need absolute coordinate.")
            print("------------------------------------------------")
        if self.use_mask and GT_MASKS in data_item:
            if not self.test_unstatu(WTransform.HWN_MASK):
                print("------------------------------------------------")
                print(f"ERROR: {self} need NHW format mask.")
                print("------------------------------------------------")
        if not self.enable:
            return data_item
        is_rotate = tf.less(tf.random_uniform(shape=[]),self.rotate_probability)
        if self.max_angle is not None:
            angle = tf.random_uniform(shape=(),minval=-self.max_angle,maxval=self.max_angle)
        else:
            angle = tf.random_uniform(shape=(), minval=-180,maxval=180)

        r_image = tfop.tensor_rotate(image=data_item[IMAGE],angle=angle)
        WTransform.cond_set(data_item, IMAGE, is_rotate, r_image)

        if GT_MASKS in data_item and self.use_mask:
            r_mask,r_bboxes = tfop.mask_rotate(mask=data_item[GT_MASKS],angle=angle,get_bboxes_stride=4)
            #r_bboxes = tf.Print(r_bboxes,["rbbox",tf.shape(r_bboxes)],summarize=100)
            WTransform.cond_set(data_item,GT_MASKS,is_rotate,r_mask)
            WTransform.cond_set(data_item, GT_BOXES, is_rotate, r_bboxes)
        else:
            if GT_MASKS in data_item:
                r_mask, _ = tfop.mask_rotate(mask=data_item[GT_MASKS], angle=angle, get_bboxes_stride=4)
                WTransform.cond_set(data_item,GT_MASKS,is_rotate,r_mask)
            if GT_BOXES in data_item:
                img_shape = tf.shape(data_item[IMAGE])
                r_bboxes = tfop.bboxes_rotate(bboxes=data_item[GT_BOXES],angle=angle,
                                           img_size = img_shape,
                                           type=self.rotate_bboxes_type)
                r_bboxes = tf.maximum(r_bboxes,0)
                max_value = tf.convert_to_tensor([[img_shape[0]-1,img_shape[1]-1,img_shape[0]-1,img_shape[1]-1]])
                max_value = tf.cast(max_value,tf.float32)
                r_bboxes = tf.minimum(r_bboxes,max_value)
                WTransform.cond_set(data_item, GT_BOXES, is_rotate, r_bboxes)
        if GT_KEYPOINTS in data_item:
            img_shape = tf.shape(data_item[IMAGE])
            rotated_points = kp.keypoits_rotate(data_item[GT_KEYPOINTS],angle,width=img_shape[1],height=img_shape[0])
            WTransform.cond_set(data_item, GT_KEYPOINTS, is_rotate, rotated_points)

        return data_item
    
    def __repr__(self):
        return f"{RandomRotateAnyAngle.__name__}"
'''
mask:H,W,N format
bbox: relative coordinate
'''
class ResizeShortestEdge(WTransform):
    def __init__(self,short_edge_length=range(640,801,32),align=1,resize_method=tf.image.ResizeMethod.BILINEAR,
                 max_size=-1,
                 seed=None):
        if isinstance(short_edge_length, Iterable):
            short_edge_length = list(short_edge_length)
        elif not isinstance(short_edge_length, list):
            short_edge_length = [short_edge_length]
        self.short_edge_length = short_edge_length
        self.align = align
        self.resize_method = resize_method
        self.seed = seed
        self.max_size = max_size
        

    def __call__(self, data_item):
        if not self.test_unstatu(WTransform.ABSOLUTE_COORDINATE):
            print(f"WARNING: {self} need relative coordinate.")
        if not self.test_statu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need HWN format mask.")
        with tf.name_scope("resize_shortest_edge"):
            if len(self.short_edge_length) == 1:
                func = partial(resize_img,limit=[self.short_edge_length[0],self.max_size],
                                  align=self.align,
                                  resize_method=self.resize_method)

            else:
                idx = tf.random_uniform(shape=(),
                                        minval=0,maxval=len(self.short_edge_length),
                                        dtype=tf.int32,
                                        seed=self.seed)
                s = tf.gather(self.short_edge_length,idx)
                func = partial(resize_img,limit=[s,self.max_size],
                               align=self.align,
                               resize_method=self.resize_method)
            return self.apply_to_images_and_masks(func,data_item)
        
    def __repr__(self):
        return type(self).__name__+f"[{self.short_edge_length}, max_size={self.max_size}]"
'''
mask:H,W,N format
'''
class ResizeLongestEdge(WTransform):
    def __init__(self,long_edge_length=range(640,801,32),align=1,resize_method=tf.image.ResizeMethod.BILINEAR,
                 min_size=-1,
                 seed=None):
        if isinstance(long_edge_length, Iterable):
            long_edge_length = list(long_edge_length)
        elif not isinstance(long_edge_length, list):
            long_edge_length = [long_edge_length]
        self.long_edge_length = long_edge_length
        self.align = align
        self.resize_method = resize_method
        self.seed = seed
        self.min_size = min_size
        

    def __call__(self, data_item):
        if not self.test_statu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need HWN format mask.")
        with tf.name_scope("resize_shortest_edge"):
            if len(self.long_edge_length) == 1:
                func = partial(resize_img,limit=[self.min_size,self.long_edge_length[0]],
                               align=self.align,
                               resize_method=self.resize_method)

            else:
                idx = tf.random_uniform(shape=(),
                                        minval=0,maxval=len(self.long_edge_length),
                                        dtype=tf.int32,
                                        seed=self.seed)
                s = tf.gather(self.long_edge_length,idx)
                func = partial(resize_img,limit=[self.min_size,s],
                               align=self.align,
                               resize_method=self.resize_method)
            return self.apply_to_images_and_masks(func,data_item)
        
    def __repr__(self):
        return f"{type(self).__name__}"

class MaskNHW2HWN(WTransform):
    def __init__(self):
        pass

    def __call__(self, data_item):
        if not self.test_unstatu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need NHW format mask.")
        self.set_statu(WTransform.HWN_MASK)
        with tf.name_scope("MaskNHW2HWN"):
            func = partial(tf.transpose,perm=[1,2,0])
            return self.apply_to_masks(func,data_item)
        
    def __repr__(self):
        return f"{type(self).__name__}"

class MaskHWN2NHW(WTransform):
    def __init__(self):
        pass

    def __call__(self, data_item):
        if not self.test_statu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need HWN format mask.")
        self.unset_statu(WTransform.HWN_MASK)
        with tf.name_scope("MaskHWN2NHW"):
            func = partial(tf.transpose,perm=[2,0,1])
            return self.apply_to_masks(func,data_item)
        
    def __repr__(self):
        return f"{type(self).__name__}"

class DelHeightWidth(WTransform):
    def __call__(self, data_item):
        if 'height' in data_item:
            del data_item['height']
        if 'width' in data_item:
            del data_item['width']
        return data_item
    def __repr__(self):
        return f"{type(self).__name__}"

class UpdateHeightWidth(WTransform):
    def __call__(self, data_item):
        shape = tf.shape(data_item[IMAGE])
        if HEIGHT in data_item:
            data_item[HEIGHT] = shape[0]
        if WIDTH in data_item:
            data_item[WIDTH] = shape[1]
        return data_item
    def __repr__(self):
        return f"{type(self).__name__}"

class AddSize(WTransform):
    def __init__(self,img_key='image'):
        self.img_key = img_key
    def __call__(self, data_item):
        assert len(data_item[self.img_key].get_shape())<=3,"error image dims size."
        data_item['size'] = tf.shape(data_item[self.img_key])[0:2]
        return data_item
    def __repr__(self):
        return f"{type(self).__name__}"

class BBoxesRelativeToAbsolute(WTransform):
    def __init__(self,img_key='image'):
        self.img_key = img_key

    def __call__(self, data_item):
        if not self.test_unstatu(WTransform.ABSOLUTE_COORDINATE):
            print(f"WARNING: {self} need relative coordinate.")
        self.set_statu(WTransform.ABSOLUTE_COORDINATE)
        with tf.name_scope("BBoxesRelativeToAbsolute"):
            size = tf.shape(data_item[self.img_key])
            if GT_BOXES in data_item:
                func = partial(wml_bboxes.tfrelative_boxes_to_absolutely_boxes,width=size[1],height=size[0])
                data_item = self.apply_to_bbox(func,data_item)
            if GT_KEYPOINTS in data_item:
                func = partial(kp.keypoints_relative2absolute, width=size[1], height=size[0])
                data_item = self.apply_to_keypoints(func, data_item)
            return data_item

    def __repr__(self):
        return f"{type(self).__name__}"

'''
prossed batched data
bboxes: [batch_size,N,4] / [N,4]
'''
class BBoxesAbsoluteToRelative(WTransform):
    def __init__(self,img_key='image'):
        self.img_key = img_key

    def __call__(self, data_item):
        if not self.test_statu(WTransform.ABSOLUTE_COORDINATE):
            print(f"WARNING: {self} need absolute coordinate.")
        self.unset_statu(WTransform.ABSOLUTE_COORDINATE)
        with tf.name_scope("BBoxesRelativeToAbsolute"):
            if len(data_item[IMAGE].get_shape()) == 4:
                size = tf.shape(data_item[self.img_key])[1:3]
            else:
                size = tf.shape(data_item[self.img_key])[:2]
            if GT_BOXES in data_item:
                func = partial(wml_bboxes.tfabsolutely_boxes_to_relative_boxes,width=size[1],height=size[0])
                data_item = self.apply_to_bbox(func,data_item)
            if GT_KEYPOINTS in data_item:
                func = partial(kp.keypoints_absolute2relative, width=size[1], height=size[0])
                data_item = self.apply_to_keypoints(func, data_item)

        return data_item

    def __repr__(self):
        return f"{type(self).__name__}"

'''
如果有Mask分支，Mask必须先转换为HWN的模式
'''
class ResizeToFixedSize(WTransform):
    def __init__(self,size=[224,224],resize_method=tf.image.ResizeMethod.BILINEAR,channels=3):
        self.resize_method = resize_method
        self.size = list(size)
        self.channels = channels

    def __call__(self, data_item):
        if not self.test_statu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need HWN format mask.")
        with tf.name_scope("resize_image"):
            def func(image):
                func0 = partial(tf.image.resize_images,size=self.size, method=self.resize_method)
                return func0(image)
                #return tf.cond(tf.reduce_any(tf.equal(tf.shape(image),0)),lambda :image,
                                 #lambda:func0(image));
            print("SIZE:",self.size)
            items = self.apply_to_images_and_masks(func,data_item)
            if self.channels is not None:
                shape = self.size+[self.channels]
                if len(items[IMAGE].shape)==4:
                    shape = [items[IMAGE].shape[0]]+shape
                func = partial(tf.reshape,shape=shape)
                items = self.apply_to_images(func,items)
            return items
        
    def __repr__(self):
        return f"{type(self).__name__}"

class NoTransform(WTransform):
    def __init__(self,id=None):
        self.id = id
    def __call__(self, data_item):
        if self.id is None:
            return data_item
        else:
            data_item[IMAGE] = tf.Print(data_item[IMAGE],["id:",self.id])
            return data_item
    def __repr__(self):
        return f"{type(self).__name__}"

class RandomSelectSubTransform(WTransform):
    def __init__(self,trans,probs=None):
        self.trans = [WTransformList(v) if isinstance(v,Iterable) else v for v in trans]
        if probs is None:
            self.probs = [1.0/len(trans)]*len(trans)
        else:
            self.probs = probs

    def __call__(self, data_item:dict):
        all_funs = []
        for t in self.trans:
            all_funs.append(partial(t,dict(data_item)))

        index = tf.random_uniform(shape=(),maxval=len(self.trans),dtype=tf.int32)

        selected_data = btf.selectfn_in_list(all_funs,index)
        return selected_data

    @staticmethod
    def is_same(datas,ref_data):
        for d in datas:
            if d != ref_data:
                return False
        return True
    def __repr__(self):
        return f"{type(self).__name}: "+str(self.trans)

class AddBoxLens(WTransform):
    def __init__(self,box_key=GT_BOXES,gt_len_key=GT_LENGTH):
        self.box_key = box_key
        self.gt_len_key = gt_len_key

    def __call__(self, data_item):
        gt_len = tf.shape(data_item[self.box_key])[0]
        data_item[self.gt_len_key] = gt_len
        return data_item
    def __repr__(self):
        return f"{type(self).__name__}"

class WDistortColor(WTransform):
    def __init__(self,color_ordering=2,**kwargs):
        self.color_ordering = color_ordering
        self.kwargs = kwargs
        
    def __call__(self, data_item):
        func = partial(distort_color,color_ordering=self.color_ordering,**self.kwargs)
        return self.apply_to_images(func,data_item)
    def __repr__(self):
        return f"{type(self).__name__}"

class WRandomDistortColor(WTransform):
    def __init__(self,probability=0.5,color_ordering=2,**kwargs):
        self.kwargs = kwargs
        self.probability = probability
        self.color_ordering = color_ordering
        
    def __call__(self, data_item):
        def fn():
            return distort_color(image=data_item[IMAGE],
                                 color_ordering=self.color_ordering)
        def fn2():
            return tf.cast(data_item[IMAGE],tf.float32)
        self.probability_fn_set(data_item,IMAGE,self.probability,
                                       fn,fn2)
        return data_item
    
    def __repr__(self):
        return f"{type(self).__name__}"

class WRandomMotionBlur(WTransform):
    def __init__(self,probability=0.5,degree=10, angle=180,**kwargs):
        self.kwargs = kwargs
        self.probability = probability
        self.degree = degree
        self.max_angle = angle
        
    def __call__(self, data_item):
        def fn():
            img = data_item[IMAGE]
            degree = tf.random_uniform(shape=(),minval=3,maxval=self.degree+1,dtype=tf.int32)
            angle = tf.random_uniform(shape=(),minval=-self.max_angle,maxval=self.max_angle,dtype=tf.int32)
            return tf.py_func(motion_blur,[img,degree,angle],dtype=img.dtype)
        self.probability_fn_set(data_item,IMAGE,self.probability,
                                       fn)
        return data_item
    
    def __repr__(self):
        return f"{type(self).__name__}"

class WRandomBlur(WTransform):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
    def __call__(self, data_item):
        return self.apply_to_images(random_blur,data_item)
    def __repr__(self):
        return f"{type(self).__name__}"

class WTransImgToFloat(WTransform):
    def __init__(self,**kwargs):
        self.kwargs = kwargs

    def __call__(self, data_item):
        data_item[IMAGE] = tf.cast(data_item[IMAGE],tf.float32)
        return data_item

    def __repr__(self):
        return f"{type(self).__name__}"

class WTransImgToGray(WTransform):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
    def __call__(self, data_item):
        return self.apply_to_images(tf.image.rgb_to_grayscale,data_item)
    def __repr__(self):
        return f"{type(self).__name__}"

class WPerImgStandardization(WTransform):
    def __init__(self,**kwargs):
        self.kwargs = kwargs

    def __call__(self, data_item):
        return self.apply_to_images(tf.image.per_image_standardization,data_item)
    def __repr__(self):
        return f"{type(self).__name__}"

class WRemoveOverlap(WTransform):
    def __init__(self,threshold=0.5,**kwargs):
        self.threshold = threshold
        self.kwargs = kwargs

    def __call__(self, data_item):
        bboxes,keep_pos = odb.remove_bboxes_by_overlap(data_item[GT_BOXES],data_item[GT_LABELS],
                                                       threshold=self.threshold)
        data_item[GT_BOXES] = bboxes
        data_item[GT_LABELS] = tf.boolean_mask(data_item[GT_LABELS],keep_pos)
        if GT_MASKS in data_item:
            data_item[GT_MASKS] = tf.boolean_mask(data_item[GT_MASKS], keep_pos)
        if GT_KEYPOINTS in data_item:
            data_item[GT_KEYPOINTS] = tf.boolean_mask(data_item[GT_KEYPOINTS], keep_pos)

        return data_item

    def __repr__(self):
        return f"{type(self).__name__}"

class RandomCrop(WTransform):
    def __init__(self,
                 crop_size=(224,224),
                 probability=None,
                 filter_threshold=0.3,
                 ):
        self.crop_size = crop_size
        self.probability = probability
        self.filter_threshold = filter_threshold

    def __call__(self, data_item):
        if not self.test_unstatu(WTransform.ABSOLUTE_COORDINATE):
            print(f"WARNING: {self} need relative coordinate.")
        if not self.test_statu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need HWN format mask.")
        #return data_item
        labels = data_item[GT_LABELS]
        data_item[GT_LABELS] = labels
        image = data_item[IMAGE]
        bboxes = data_item[GT_BOXES]
        if self.probability is not None and self.probability<1:
            distored = tf.less_equal(tf.random_uniform(shape=()),self.probability)
        else:
            distored = None
        dst_image, labels, bboxes, bbox_begin,bbox_size,bboxes_mask= \
            random_crop(image, labels, bboxes,
                        crop_size=self.crop_size,
                        filter_threshold=self.filter_threshold,
                        scope="random_crop_bboxes")
        self.cond_set(data_item,IMAGE,distored,dst_image)
        self.cond_set(data_item,GT_LABELS,distored,labels)
        self.cond_set(data_item,GT_BOXES,distored,bboxes)

        if GT_MASKS in data_item:
            mask = data_item[GT_MASKS]
            mask = tf.transpose(mask,perm=[2,0,1])
            mask = tf.boolean_mask(mask,bboxes_mask)
            mask = tf.transpose(mask,perm=[1,2,0])
            mask = tf.slice(mask,bbox_begin,bbox_size)
            self.cond_set(data_item, GT_MASKS, distored, mask)
            pass
        if GT_KEYPOINTS in data_item:
            print(f"WARNING: keypoints don't support transform {self}.")

        return data_item

    def __repr__(self):
        return f"{type(self).__name__}"
'''
bboxes: relative coordinate
如果有Mask分支，Mask必须先转换为HWN的模式
'''
class SampleDistortedBoundingBox(WTransform):
    def __init__(self,
                 min_object_covered=0.3,
                 aspect_ratio_range=(0.9, 1.1),
                 area_range=(0.1, 1.0),
                 max_attempts=100,
                 filter_threshold=0.3,
                 use_image_if_no_bounding_boxes=True,
                 ):
        self.min_object_covered = min_object_covered
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts
        self.filter_threshold = filter_threshold
        self.use_image_if_no_bounding_boxes = use_image_if_no_bounding_boxes

    def __call__(self, data_item):
        if not self.test_unstatu(WTransform.ABSOLUTE_COORDINATE):
            print(f"WARNING: {self} need relative coordinate.")
        if not self.test_statu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need HWN format mask.")

        print(f"{self} area range {self.area_range}, aspect reatio {self.aspect_ratio_range}.")
            
        labels = data_item[GT_LABELS]
        data_item[GT_LABELS] = labels
        image = data_item[IMAGE]
        bboxes = data_item[GT_BOXES]
        dst_image, labels, bboxes, bbox_begin,bbox_size,bboxes_mask= \
            distorted_bounding_box_crop(image, labels, bboxes,
                                        area_range=self.area_range,
                                        min_object_covered=self.min_object_covered,
                                        aspect_ratio_range=self.aspect_ratio_range,
                                        max_attempts=self.max_attempts,
                                        filter_threshold=self.filter_threshold,
                                        use_image_if_no_bounding_boxes=self.use_image_if_no_bounding_boxes,
                                        scope="distorted_bounding_box_crop")
        data_item[IMAGE] = dst_image
        data_item[GT_LABELS] = labels
        data_item[GT_BOXES] = bboxes

        if GT_MASKS in data_item:
            def func(mask):
                mask = tf.transpose(mask,perm=[2,0,1])
                mask = tf.boolean_mask(mask,bboxes_mask)
                mask = tf.transpose(mask,perm=[1,2,0])
                mask = tf.slice(mask,bbox_begin,bbox_size)
                return mask
            data_item = self.apply_to_masks(func,data_item)
        if GT_KEYPOINTS in data_item:
            print(f"WARNING: keypoints don't support transform {self}.")

        return data_item
    
    def __repr__(self):
        return f"{type(self).__name__}"

class RandomSampleDistortedBoundingBox(WTransform):
    def __init__(self,
                 min_object_covered=0.3,
                 aspect_ratio_range=(0.9, 1.1),
                 area_range=(0.1, 1.0),
                 max_attempts=100,
                 filter_threshold=0.3,
                 probability=0.5,
                 ):
        self.min_object_covered = min_object_covered
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts
        self.filter_threshold = filter_threshold
        self.probability = probability

    def __call__(self, data_item):
        if not self.test_unstatu(WTransform.ABSOLUTE_COORDINATE):
            print(f"WARNING: {self} need relative coordinate.")
        if not self.test_statu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need HWN format mask.")
        #return data_item
        labels = data_item[GT_LABELS]
        data_item[GT_LABELS] = labels
        image = data_item[IMAGE]
        bboxes = data_item[GT_BOXES]
        distored = tf.less_equal(tf.random_uniform(shape=()),self.probability)
        dst_image, labels, bboxes, bbox_begin,bbox_size,bboxes_mask= \
            distorted_bounding_box_crop(image, labels, bboxes,
                                        area_range=self.area_range,
                                        min_object_covered=self.min_object_covered,
                                        aspect_ratio_range=self.aspect_ratio_range,
                                        max_attempts=self.max_attempts,
                                        filter_threshold=self.filter_threshold,
                                        scope="distorted_bounding_box_crop")
        self.cond_set(data_item,IMAGE,distored,dst_image)
        self.cond_set(data_item,GT_LABELS,distored,labels)
        self.cond_set(data_item,GT_BOXES,distored,bboxes)

        if GT_MASKS in data_item:
            mask = data_item[GT_MASKS]
            mask = tf.transpose(mask,perm=[2,0,1])
            mask = tf.boolean_mask(mask,bboxes_mask)
            mask = tf.transpose(mask,perm=[1,2,0])
            mask = tf.slice(mask,bbox_begin,bbox_size)
            self.cond_set(data_item, GT_MASKS, distored, mask)
            pass
        if GT_KEYPOINTS in data_item:
            print(f"WARNING: keypoints don't support transform {self}.")

        return data_item
    
    def __repr__(self):
        return f"{type(self).__name__}"

class WTransLabels(WTransform):
    def __init__(self,id_to_label):
        '''
        原有的label称为id, 变换后的称为label
        id_to_label为一个字典, key为id, value为label
        '''
        self.id_to_label = id_to_label

    def __call__(self, data_item):
        labels = data_item[GT_LABELS]
        labels = tfop.int_hash(labels,self.id_to_label)
        data_item[GT_LABELS] = labels
        return data_item
    def __repr__(self):
        return f"{type(self).__name__}"

'''
Mask必须为NHW格式
'''
class WRemoveCrowdInstance(WTransform):
    def __init__(self,remove_crowd=True):
        self.remove_crowd = remove_crowd

    def __call__(self, data_item):

        if (IS_CROWD not in data_item) or not self.remove_crowd:
            return data_item
        with tf.name_scope("remove_crowd_instance"):
            is_crowd = data_item[IS_CROWD]
            indices = tf.squeeze(tf.where(tf.equal(is_crowd,0)),axis=-1)
            labels = tf.gather(data_item[GT_LABELS],indices)
            bboxes = tf.gather(data_item[GT_BOXES],indices)
            is_crowd = tf.gather(data_item[IS_CROWD],indices)
            data_item[GT_BOXES] = bboxes
            data_item[GT_LABELS] = labels
            data_item[IS_CROWD] = is_crowd
            if GT_MASKS in data_item:
                masks = tf.gather(data_item[GT_MASKS],indices)
                data_item[GT_MASKS] = masks
            if GT_KEYPOINTS in data_item:
                keypoints = tf.gather(data_item[GT_KEYPOINTS], indices)
                data_item[GT_KEYPOINTS] = keypoints

        return data_item

    def __repr__(self):
        return f"{type(self).__name__}"
'''
Mask必须为HWN格式
'''
class WScale(WTransform):
    def __init__(self,scale=0.5,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
        self.scale = scale
        self.method = method

    def __call__(self, data_item):
        if not self.test_statu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need HWN format mask.")

        image = wnnl.upsample(data_item[IMAGE],scale_factor=self.scale,mode=self.method)
        data_item[IMAGE] = image
        if GT_MASKS in data_item:
            mask = wnnl.upsample(data_item[GT_MASKS],scale_factor=self.scale,mode=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            data_item[GT_MASKS] = mask

        return data_item
    def __repr__(self):
        return f"{type(self).__name__}"
'''
operator on batch
'''
class FixDataInfo(WTransform):
    def __init__(self,channel=3):
        self.channel = channel

    def __call__(self, data_item):
        def func(x):
            batch_size,H,W,C = btf.combined_static_and_dynamic_shape(x)
            return tf.reshape(x,[batch_size,H,W,self.channel])
        return self.apply_to_images(func,data_item)
    def __repr__(self):
        return f"{type(self).__name__}"
'''
operator on batch
'''
class CheckBBoxes(WTransform):
    def __init__(self,min=0,max=1):
        self.min = min
        self.max = max

    def __call__(self, data_item):
        if GT_BOXES in data_item:
            data_item[GT_BOXES] = tf.clip_by_value(data_item[GT_BOXES],self.min,self.max)
        if GT_KEYPOINTS in data_item:
            data_item[GT_KEYPOINTS] = tf.clip_by_value(data_item[GT_KEYPOINTS],-2,self.max) #keypoints use -1 to indict unlabeled points
        return data_item

    def __repr__(self):
        return f"{type(self).__name__}"

'''
image: [B,H,W,C]
bboxes: absolute coordinate
如果有Mask分支，Mask必须为[B,N,H,W]
'''
class PadtoAlign(WTransform):
    def __init__(self,align=1):
        self.align = align

    def __call__(self, data_item):
        if not self.test_statu(WTransform.ABSOLUTE_COORDINATE):
            print(f"WARNING: {self} need absolute coordinate.")
        if not self.test_unstatu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need NHW format mask.")
        if self.align <= 1:
            return data_item

        def get_pad_value(v):
            return ((v+self.align-1)//self.align)*self.align-v


        def func4img(x):
            batch_size,H,W,C = btf.combined_static_and_dynamic_shape(x)
            padd_H = get_pad_value(H)
            padd_W = get_pad_value(W)
            return tf.pad(x,paddings=[[0,0,],[0,padd_H],[0,padd_W],[0,0]])

        def func4mask(x):
            batch_size, N,H, W = btf.combined_static_and_dynamic_shape(x)
            padd_H = get_pad_value(H)
            padd_W = get_pad_value(W)
            return tf.pad(x, paddings=[[0, 0, ], [0,0],[0, padd_H], [0, padd_W]])

        data_item = self.apply_to_images(func4img,data_item)
        return self.apply_to_masks(func4mask,data_item)
    
    def __repr__(self):
        return f"{type(self).__name__}"
'''
image: [B,H,W,C]
bboxes: absolute coordinate
如果有Mask分支，Mask必须为[B,N,H,W]
'''
class PadtoFixedSize(WTransform):
    def __init__(self,size=1):
        '''
        size: [H,W]
        '''
        self.size = size 

    def __call__(self, data_item):
        if not self.test_statu(WTransform.ABSOLUTE_COORDINATE):
            print(f"WARNING: {self} need absolute coordinate.")
        if not self.test_unstatu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need NHW format mask.")
        if self.align <= 1:
            return data_item

        def get_padh_value(v):
            return self.size[0]-v
        def get_padw_value(v):
            return self.size[1]-v


        def func4img(x):
            batch_size,H,W,C = btf.combined_static_and_dynamic_shape(x)
            padd_H = get_padh_value(H)
            padd_W = get_padw_value(W)
            return tf.pad(x,paddings=[[0,0,],[0,padd_H],[0,padd_W],[0,0]])

        def func4mask(x):
            batch_size, N,H, W = btf.combined_static_and_dynamic_shape(x)
            padd_H = get_padh_value(H)
            padd_W = get_padw_value(W)
            return tf.pad(x, paddings=[[0, 0, ], [0,0],[0, padd_H], [0, padd_W]])

        data_item = self.apply_to_images(func4img,data_item)
        return self.apply_to_masks(func4mask,data_item)
    
    def __repr__(self):
        return f"{type(self).__name__}"

'''
用于将多个图像拼在一起，生成一个包含更多（通常也更清晰）的小目标
image: [B,H,W,C]
如果有Mask分支，Mask必须为[B,N,H,W]
bboxes:absolute coordinate
'''
class Stitch(WTransform):
    def __init__(self,nr=1):
        if nr<1:
            self.nr = 1
            self.probability = nr
        else:
            self.nr = nr
            self.probability = None
            

    def __call__(self, data_item):
        if not self.test_statu(WTransform.ABSOLUTE_COORDINATE):
            print(f"WARNING: {self} need absolute coordinate.")
        if not self.test_unstatu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need NHW format mask.")
        B,H,W,C = btf.combined_static_and_dynamic_shape(data_item[IMAGE])
        if B<2:
            return data_item

        if self.probability is None:
            return self.apply_stitch(data_item)
        else:
            p = tf.less(tf.random_uniform(shape=()),self.probability)
            return tf.cond(p,partial(self.apply_stitch,dict(data_item)),lambda:dict(data_item))

    def apply_stitch(self, data_item):
        B,H,W,C = btf.combined_static_and_dynamic_shape(data_item[IMAGE])
        if B<2:
            return data_item

        indexs = tf.range((B//2)*2)
        indexs = tf.random_shuffle(indexs,seed=time.time())
        indexs = tf.reshape(indexs,[B//2,2])
        nr = tf.minimum(B//2,int(self.nr+0.5))
        indexs = indexs[:nr]
        B,H,W,C = btf.combined_static_and_dynamic_shape(data_item[IMAGE])
        B,_,BOX_DIM = btf.combined_static_and_dynamic_shape(data_item[GT_BOXES])
        indexs = tf.transpose(indexs, [1, 0])
        o_indexs = indexs[0]
        #get maxlength
        l0 = tf.gather(data_item[GT_LENGTH],indexs[0])
        l1 = tf.gather(data_item[GT_LENGTH],indexs[1])
        sum_l = l0+l1
        max_length = tf.maximum(tf.reduce_max(sum_l),tf.shape(data_item[GT_LABELS])[1])
        pad_value = tf.maximum(0,max_length-tf.shape(data_item[GT_LABELS])[1])
        WTransform.pad(data_item,GT_BOXES,[[0,0],[0,pad_value]])
        WTransform.pad(data_item,GT_LABELS,[[0,0],[0,pad_value]])
        if GT_MASKS in data_item:
            WTransform.pad(data_item,GT_MASKS,[[0,0],[0,pad_value]])
        #
        indexs = tf.reshape(indexs, [-1])

        i_image = tf.gather(data_item[IMAGE],indexs)
        i_bboxes = tf.gather(data_item[GT_BOXES],indexs)
        i_labels = tf.gather(data_item[GT_LABELS],indexs)
        i_width = tf.gather(data_item[WIDTH],indexs)
        i_height = tf.gather(data_item[HEIGHT],indexs)
        i_length = tf.gather(data_item[GT_LENGTH],indexs)

        def set_value(data_dict,key,value,indexs):
            org_v = data_dict[key]
            shape = btf.combined_static_and_dynamic_shape(org_v)
            cond_v = tf.scatter_nd(tf.reshape(indexs,[-1,1]),value,shape)
            is_set = tf.cast(btf.indices_to_dense_vector(indexs,size=shape[0]),tf.bool)
            cond_v = tf.where(is_set,cond_v,org_v)
            data_dict[key] = cond_v
            pass

        if GT_MASKS in data_item:
            mask = tf.gather(data_item[GT_MASKS],indexs)
            mask = tf.transpose(mask,perm=[0,2,3,1])
            image, mask, bboxes, labels,length = tf.py_func(Stitch.process_with_masks,[i_image,
                                                                                           mask,
                                                                                           i_bboxes,
                                                                                           i_labels,
                                                                                           i_width,
                                                                                           i_height,
                                                                                           i_length],
                                                           (data_item[IMAGE].dtype,
                                                            data_item[GT_MASKS].dtype,
                                                            tf.float32,
                                                            data_item[GT_LABELS].dtype,
                                                            data_item[GT_LENGTH].dtype),
                                                            stateful=False)
            mask = tf.transpose(mask,perm=[0,3,1,2])
            mask = tf.reshape(mask, [nr,max_length,H,W])
            set_value(data_item,GT_MASKS,mask,o_indexs)
        else:
            image, bboxes, labels,length = tf.py_func(Stitch.process,[i_image,
                                                                              i_bboxes,
                                                                              i_labels,
                                                                              i_width,
                                                                              i_height,
                                                                              i_length],
                                                           (data_item[IMAGE].dtype,
                                                            tf.float32,
                                                            data_item[GT_LABELS].dtype,
                                                            data_item[GT_LENGTH].dtype),
                                                            stateful=False)

        image = tf.reshape(image, [nr,H,W,C])
        bboxes = tf.reshape(bboxes, [nr, max_length, BOX_DIM])
        labels = tf.reshape(labels, [nr, max_length])
        length = tf.reshape(length, [nr])
        set_value(data_item, IMAGE,  image,o_indexs)
        set_value(data_item, GT_BOXES,  bboxes,o_indexs)
        set_value(data_item, GT_LABELS,  labels,o_indexs)
        set_value(data_item, GT_LENGTH,  length,o_indexs)

        return data_item

    @staticmethod
    def fitto(width,height,target_w,target_h):
        '''
        用于计算将大小为(height,width)的区域等比例缩放到(target_h,target_w)的大小的区域时的
        准确尺寸
        :param width:
        :param height:
        :param target_w:
        :param target_h:
        :return:
        '''
        if width<1 or height<1:
            return (target_w,target_h)
        if width*target_h>=height*target_w:
            res_w = target_w
            res_h = int(height* res_w / width)
        else:
            res_h=target_h
            res_w = int(width*res_h/height)

        return (res_w,res_h)

    @staticmethod
    def is_h(W,H,widths,heights):
        TW = W
        TH = H/2
        ws = []
        hs = []
        for w,h in zip(widths,heights):
            w0,h0 = Stitch.fitto(w,h,TW,TH)
            ws.append(w0)
            hs.append(h0)
        int_h = W*H-ws[0]*hs[0]-ws[1]*hs[1]
        TW = W//2
        TH = H
        ws = []
        hs = []
        for w,h in zip(widths,heights):
            w0,h0 = Stitch.fitto(w,h,TW,TH)
            ws.append(w0)
            hs.append(h0)
        int_v = W*H-ws[0]*hs[0]-ws[1]*hs[1]

        return int_h<=int_v

    @staticmethod
    def get_sizes(W,H,widths,heights,is_h):
        '''
        多个区域将以横向(is_h=True)或纵向(is_h=False)的拼接方式放在大小为(H,W)的区域时，他们的目标大小
        :param W: ()
        :param H: ()
        :param widths: [N]
        :param heights:  [N]
        :param is_h: ()
        :return:
        [N],[N]
        '''
        if is_h:
            TW = W
            TH = H//2
        else:
            TW = W//2
            TH = H

        ws = []
        hs = []
        for w,h in zip(widths,heights):
            w0,h0 = Stitch.fitto(w,h,TW,TH)
            ws.append(w0)
            hs.append(h0)
        return ws,hs

    @staticmethod
    def get_scales(W,H,TW,TH):
        return TW/W,TH/H

    @staticmethod
    def concat_img(imgs,widths,heights,W,H,is_h):
        ws,hs = Stitch.get_sizes(W,H,widths,heights,is_h)
        timgs = []
        for i,img,width,height,tw,th in zip(count(),imgs,widths,heights,ws,hs):
            img = img[:height,:width]
            img = wmli.resize_img(img,(tw,th),keep_aspect_ratio=False)
            timgs.append(img)
        if is_h:
            timgs = Stitch.pad_imgs_to(timgs,[[H//2,W],[H-H//2,W]])
            return np.concatenate(timgs,axis=0)
        else:
            timgs = Stitch.pad_imgs_to(timgs,[[H,W//2],[H,W-W//2]])
            return np.concatenate(timgs,axis=1)

    @staticmethod
    def pad_imgs_to(imgs,shapes):
        res = []
        for i,x,shape in zip(count(),imgs,shapes):
            res.append(np.pad(x,[(0,shape[0]-x.shape[0]),(0,shape[1]-x.shape[1]),(0,0)],'constant', constant_values=(0,0)))
        return res

    @staticmethod
    def pad_img_to(img,shape):
        return np.pad(img,[(0,shape[0]-img.shape[0]),(0,shape[1]-img.shape[1]),(0,0)],'constant', constant_values=(0,0))

    @staticmethod
    def concat_mask(imgs,widths,heights,W,H,is_h):
        '''

        :param imgs: [[H,W,N0],[H,W,N1]]
        :param widths: [w0,w1]
        :param heights: [h0,h1]
        :param W: ()
        :param H: ()
        :param is_h: ()
        :return:
        '''
        ws,hs = Stitch.get_sizes(W,H,widths,heights,is_h)
        timgs = []

        if is_h:
            pad_args0 = [H//2,W]
            pad_args1 = [[(0,H-H//2),(0,0),(0,0)],
                         [(H-H // 2,0), (0, 0), (0, 0)]]
        else:
            pad_args0 = [H, W//2]
            pad_args1 = [[(0,0),(0, W - W // 2), (0, 0)],
                         [(0,0),(W - W // 2, 0), (0, 0)]]

        for i,img,width,height,tw,th in zip(count(),imgs,widths,heights,ws,hs):
            img = img[:height,:width]
            img = wmli.resize_img(img,(tw,th),keep_aspect_ratio=False)
            if len(img.shape) == 2:
                img = np.expand_dims(img,axis=-1)
            img = Stitch.pad_img_to(img,pad_args0)
            img = np.pad(img,
                         pad_args1[i],
                         'constant',
                         constant_values=(0, 0))
            timgs.append(img)

        return np.concatenate(timgs,axis=2)

    @staticmethod
    def scale_bboxes(bboxes,sw,sh):
        return np.array([[sh,sw,sh,sw]])*bboxes

    @staticmethod
    def concat_bboxes(bboxes,widths,heights,W,H,is_h):
        ws,hs = Stitch.get_sizes(W,H,widths,heights,is_h)
        sw0,sh0 = Stitch.get_scales(widths[0],heights[0],ws[0],hs[0])
        sw1,sh1 = Stitch.get_scales(widths[1],heights[1],ws[1],hs[1])
        bboxes0 = Stitch.scale_bboxes(bboxes[0],sw0,sh0)
        bboxes1 = Stitch.scale_bboxes(bboxes[1],sw1,sh1)

        if is_h:
            offset = [0,H//2]
        else:
            offset = [W//2,0]
        bboxes1 = bboxes1+np.array([[offset[1],offset[0],offset[1],offset[0]]])
        return np.concatenate([bboxes0,bboxes1],axis=0)

    @staticmethod
    def concat_labels(labels):
        return np.concatenate(labels,axis=0)

    @staticmethod
    def __process(image,mask,bboxes,labels,width,height,length):
        B,H,W = image.shape[:3]
        res_nr = B//2
        indexs = np.array(range(B))
        indexs = np.transpose(np.reshape(indexs,[2,-1]),axes=[1,0])
        max_length = bboxes.shape[1]
        res_length = np.copy(length)
        type = length.dtype
        width = np.reshape(width,[2,-1])
        height = np.reshape(height,[2,-1])
        length = np.reshape(length,[2,-1])

        for i,inds in enumerate(indexs):
            id0 = inds[0]
            id1 = inds[1]
            widths = width[:,id0]
            heights = height[:,id0]
            ih = Stitch.is_h(W,H,widths,heights)
            len0,len1 = length[:,id0]
            img = Stitch.concat_img([image[id0],image[id1]],widths,heights,W,H,ih)
            bbox = Stitch.concat_bboxes([bboxes[id0][:len0],bboxes[id1][:len1]],widths,heights,W,H,ih)
            label = Stitch.concat_labels([labels[id0][:len0],labels[id1][:len1]])
            bbox = np.pad(bbox,[(0,max_length-len0-len1),(0,0)],'constant', constant_values=(0,0))
            label = np.pad(label,[(0,max_length-len0-len1)],'constant', constant_values=(0,0))
            res_length[id0] = len0+len1
            image[id0] = img
            bboxes[id0] = bbox
            labels[id0] = label
            try:
                if mask is not None:
                    m = Stitch.concat_mask([mask[id0][:,:,:len0],mask[id1][:,:,:len1]],widths,heights,W,H,ih)
                    m = np.pad(m,[(0,0),(0,0),(0,max_length-len0-len1)],'constant', constant_values=(0,0))
                    mask[id0] = m
            except:
                raise 1
                pass

        if mask is not None:
            return image[:res_nr],mask[:res_nr],bboxes[:res_nr],labels[:res_nr],res_length.astype(type)[:res_nr]
        else:
            return image[:res_nr],bboxes[:res_nr],labels[:res_nr],res_length.astype(type)[:res_nr]

    @staticmethod
    def process(image,bboxes,labels,width,height,length):
        return Stitch.__process(image=image,
                                mask=None,
                                bboxes=bboxes,
                                labels=labels,
                                width=width,
                                height=height,
                                length=length)

    @staticmethod
    def process_with_masks(image,mask,bboxes,labels,width,height,length):
        return Stitch.__process(image=image,
                                mask=mask,
                                bboxes=bboxes,
                                labels=labels,
                                width=width,
                                height=height,
                                length=length)
    def __repr__(self):
        return f"{type(self).__name__}"


'''
用于将多个图像拼在一起，生成一个包含更多（通常也更清晰）的小目标
image: [B,H,W,C]
如果有Mask分支，Mask必须为[B,N,H,W]
bboxes:absolute coordinate
'''
class CopyPaste(WTransform):
    def __init__(self, nr=1):
        if nr < 1:
            self.nr = 1
            self.probability = nr
        else:
            self.nr = nr
            self.probability = None

    def __call__(self, data_item):
        if not self.test_statu(WTransform.ABSOLUTE_COORDINATE):
            print(f"WARNING: {self} need absolute coordinate.")
        if not self.test_unstatu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need NHW format mask.")
        B, H, W, C = btf.combined_static_and_dynamic_shape(data_item[IMAGE])
        if B < 2:
            return data_item

        if self.probability is None:
            return self.apply_copy_paste(data_item)
        else:
            p = tf.less(tf.random_uniform(shape=()), self.probability)
            return tf.cond(p, partial(self.apply_copy_paste, dict(data_item)), lambda: dict(data_item))

    def apply_copy_paste(self, data_item):
        B, H, W, C = btf.combined_static_and_dynamic_shape(data_item[IMAGE])
        if B < 2:
            return data_item

        indexs = tf.range((B // 2) * 2)
        indexs = tf.random_shuffle(indexs, seed=time.time())
        indexs = tf.reshape(indexs, [B // 2, 2])
        nr = tf.minimum(B // 2, int(self.nr + 0.5))
        indexs = indexs[:nr]
        B, H, W, C = btf.combined_static_and_dynamic_shape(data_item[IMAGE])
        if GT_BOXES in data_item:
            B, _, BOX_DIM = btf.combined_static_and_dynamic_shape(data_item[GT_BOXES])
        indexs = tf.transpose(indexs, [1, 0])
        o_indexs = indexs[0]
        # get maxlength
        l0 = tf.gather(data_item[GT_LENGTH], indexs[0])
        l1 = tf.gather(data_item[GT_LENGTH], indexs[1])
        sum_l = l0 + l1
        max_length = tf.maximum(tf.reduce_max(sum_l), tf.shape(data_item[GT_LABELS])[1])
        pad_value = tf.maximum(0, max_length - tf.shape(data_item[GT_LABELS])[1])
        if GT_BOXES in data_item:
            WTransform.pad(data_item, GT_BOXES, [[0, 0], [0, pad_value]])
        if GT_KEYPOINTS in data_item:
            WTransform.pad(data_item, GT_KEYPOINTS, [[0, 0], [0, pad_value]])
        WTransform.pad(data_item, GT_LABELS, [[0, 0], [0, pad_value]])
        if GT_MASKS in data_item:
            WTransform.pad(data_item, GT_MASKS, [[0, 0], [0, pad_value]])
        #
        indexs = tf.reshape(indexs, [-1])

        i_image = tf.gather(data_item[IMAGE], indexs)
        if GT_BOXES in data_item:
            i_bboxes = tf.gather(data_item[GT_BOXES], indexs)
        if GT_KEYPOINTS in data_item:
            i_kps = tf.gather(data_item[GT_KEYPOINTS], indexs)
        i_labels = tf.gather(data_item[GT_LABELS], indexs)
        i_length = tf.gather(data_item[GT_LENGTH], indexs)

        def set_value(data_dict, key, value, indexs):
            org_v = data_dict[key]
            shape = btf.combined_static_and_dynamic_shape(org_v)
            cond_v = tf.scatter_nd(tf.reshape(indexs, [-1, 1]), value, shape)
            is_set = tf.cast(btf.indices_to_dense_vector(indexs, size=shape[0]), tf.bool)
            cond_v = tf.where(is_set, cond_v, org_v)
            data_dict[key] = cond_v
            pass

        o_kps = None

        if GT_MASKS in data_item:
            mask = tf.gather(data_item[GT_MASKS], indexs)
            mask = tf.transpose(mask, perm=[0, 2, 3, 1])
            image, mask, bboxes, labels, length = tf.py_func(CopyPaste.process_with_masks, [i_image,
                                                                                         mask,
                                                                                         i_bboxes,
                                                                                         i_labels,
                                                                                         i_length],
                                                             (data_item[IMAGE].dtype,
                                                              data_item[GT_MASKS].dtype,
                                                              tf.float32,
                                                              data_item[GT_LABELS].dtype,
                                                              data_item[GT_LENGTH].dtype),
                                                             stateful=False)
            mask = tf.transpose(mask, perm=[0, 3, 1, 2])
            mask = tf.reshape(mask, [nr, max_length, H, W])
            set_value(data_item, GT_MASKS, mask, o_indexs)
        elif GT_BOXES in data_item:
            image, bboxes, labels, length = tf.py_func(CopyPaste.process, [i_image,
                                                                        i_bboxes,
                                                                        i_labels,
                                                                        i_length],
                                                       (data_item[IMAGE].dtype,
                                                        tf.float32,
                                                        data_item[GT_LABELS].dtype,
                                                        data_item[GT_LENGTH].dtype),
                                                       stateful=False)
        elif GT_KEYPOINTS in data_item:
            image, labels, o_kps,length = tf.py_func(CopyPaste.process_kps, [i_image,
                                                                       i_labels,
                                                                       i_kps,
                                                                       i_length],
                                                       (data_item[IMAGE].dtype,
                                                        data_item[GT_LABELS].dtype,
                                                        tf.float32,
                                                        data_item[GT_LENGTH].dtype),
                                                       stateful=False)
        else:
            print(f"ERROR: error data item.")

        image = tf.reshape(image, [nr, H, W, C])
        if GT_BOXES in data_item:
            bboxes = tf.reshape(bboxes, [nr, max_length, BOX_DIM])
            set_value(data_item, GT_BOXES, bboxes, o_indexs)
        if GT_KEYPOINTS in data_item and o_kps is not None:
            set_value(data_item, GT_KEYPOINTS, o_kps, o_indexs)

        labels = tf.reshape(labels, [nr, max_length])
        length = tf.reshape(length, [nr])
        set_value(data_item, IMAGE, image, o_indexs)
        set_value(data_item, GT_LABELS, labels, o_indexs)
        set_value(data_item, GT_LENGTH, length, o_indexs)

        return data_item

    @staticmethod
    def concat_mask(imgs):
        '''

        :param imgs: [[H,W,N0],[H,W,N1]]
        :return:
        '''
        return np.concatenate(imgs, axis=-1)

    @staticmethod
    def scale_bboxes(bboxes, sw, sh):
        return np.array([[sh, sw, sh, sw]]) * bboxes

    @staticmethod
    def concat_bboxes(bboxes):
        return np.concatenate(bboxes, axis=0)

    @staticmethod
    def concat_kps(kps):
        return np.concatenate(kps, axis=0)

    @staticmethod
    def concat_labels(labels):
        return np.concatenate(labels, axis=0)

    @staticmethod
    def resize_data(img,bboxes,masks,kps,min_ratio=0.5):
        OH,OW = img.shape[:2]
        new_img = np.zeros_like(img)
        ratio = np.random.rand(1)*(1-min_ratio)+min_ratio
        NH = int(OH*ratio)
        NW = int(OW*ratio)
        img = wmli.resize_img(img,(NW,NH))
        offset_x = random.randint(0,max(0,OW-NW))
        offset_y = random.randint(0,max(0,OH-NH))
        new_img[ offset_y:offset_y + img.shape[0],offset_x:offset_x + img.shape[1]] = img
        if bboxes is not None:
            bboxes = bboxes*ratio+np.array([[offset_y,offset_x,offset_y,offset_x]],dtype=bboxes.dtype)
        if kps is not None:
            if len(kps.shape)==2:
                kps = kps*ratio+np.array([[offset_x,offset_y]],dtype=kps.dtype)
            elif len(kps.shape)==3:
                kps = kps * ratio + np.array([[[offset_x, offset_y]]], dtype=kps.dtype)

        if masks is not None:
            OMH,OMW = masks.shape[:2]
            NMH = int(OMH*ratio)
            NMW = int(OMW*ratio)
            new_masks = np.zeros_like(masks)
            masks = wmli.resize_img(masks,(NMW,NMH),interpolation=cv2.INTER_NEAREST)
            mask_offset_x = int(offset_x*OMW/OW)
            mask_offset_y = int(offset_y*OMH/OH)
            new_masks[mask_offset_y:mask_offset_y+masks.shape[0],mask_offset_x:mask_offset_x+masks.shape[1]] = masks
        else:
            new_masks = None
        return new_img,bboxes,new_masks,kps

    @staticmethod
    def __process(image, mask=None, bboxes=None, labels=None, kps=None, length=None):
        B, H, W = image.shape[:3]
        res_nr = B // 2
        indexs = np.array(range(B))
        indexs = np.transpose(np.reshape(indexs, [2, -1]), axes=[1, 0])
        max_length = labels.shape[1]
        res_length = np.copy(length)
        type = length.dtype
        length = np.reshape(length, [2, -1])

        for i, inds in enumerate(indexs):
            id0 = inds[0]
            id1 = inds[1]
            len0, len1 = length[:, id0]
            bboxes_in = bboxes[id1][:len1] if bboxes is not None else None
            masks_in = mask[id0][:,:,:len1] if mask is not None else None
            kps_in = kps[id1][:len1] if kps is not None else None
            image1,bboxes1,masks1,kps1 = CopyPaste.resize_data(image[id1],bboxes_in,masks_in,kps_in
            )
            img_type = image[id0].dtype
            img = (image[id0]*0.5+image1*0.5).astype(img_type)
            image[id0] = img

            if bboxes is not None:
                bbox = CopyPaste.concat_bboxes([bboxes[id0][:len0], bboxes1])
                bbox = np.pad(bbox, [(0, max_length - len0 - len1), (0, 0)], 'constant', constant_values=(0, 0))
                bboxes[id0] = bbox

            if kps is not None:
                tkps = CopyPaste.concat_kps([kps[id0][:len0],kps1])
                tkps = np.pad(tkps, [(0, max_length - len0 - len1)]+[(0,0)]*(len(tkps.shape)-1), 'constant', constant_values=(0, 0))
                kps[id0] = tkps
            label = CopyPaste.concat_labels([labels[id0][:len0], labels[id1][:len1]])
            label = np.pad(label, [(0, max_length - len0 - len1)], 'constant', constant_values=(0, 0))
            res_length[id0] = len0 + len1
            labels[id0] = label
            try:
                if mask is not None:
                    m = CopyPaste.concat_mask([mask[id0][:, :, :len0], mask[id1][:, :, :len1]])
                    m = np.pad(m, [(0, 0), (0, 0), (0, max_length - len0 - len1)], 'constant', constant_values=(0, 0))
                    mask[id0] = m
            except:
                raise 1
                pass

        if bboxes is not None:
            res_bboxes = bboxes[:res_nr]
        else:
            res_bboxes = None

        if mask is not None:
            res_mask = mask[:res_nr]
        else:
            res_mask = None
        if kps is not None:
            res_kps = kps[:res_nr]
        else:
            res_kps = None

        return image[:res_nr], res_mask, res_bboxes,labels[:res_nr], res_kps,res_length.astype(type)[:res_nr]

    @staticmethod
    def process(image, bboxes, labels,length):
        res = CopyPaste.__process(image=image,
                                mask=None,
                                bboxes=bboxes,
                                labels=labels,
                                length=length)
        return res[0],res[2],res[3],res[5]

    @staticmethod
    def process_kps(image, labels, kps,length):
        res = CopyPaste.__process(image=image,
                                  mask=None,
                                  bboxes=None,
                                  labels=labels,
                                  kps=kps,
                                  length=length)
        return res[0],res[3],res[4],res[5]

    @staticmethod
    def process_with_masks(image, mask, bboxes, labels,kps, length):
        res = CopyPaste.__process(image=image,
                                mask=mask,
                                bboxes=bboxes,
                                labels=labels,
                                kps=kps,
                                length=length)
        return res[0],res[1],res[2],res[3],res[5]

    def __repr__(self):
        return f"{type(self).__name__}"
'''
image: in [0,255]
'''
class WRandomEqualize(WTransform):
    def __init__(self,prob=0.8):
        self.prob = prob

    def __call__(self, data_item):
        image = equalize(data_item[IMAGE])
        self.probability_set(data_item,IMAGE,self.prob,image)
        return data_item
    def __repr__(self):
        return f"{type(self).__name__}"

class WRandomCutout(WTransform):
    def __init__(self,pad_size=8,prob=0.8):
        self.prob = prob
        self.pad_size = pad_size

    def __call__(self, data_item):
        image = cutout(data_item[IMAGE],pad_size=self.pad_size)
        self.probability_set(data_item,IMAGE,self.prob,image)
        return data_item
    
    def __repr__(self):
        return f"{type(self).__name__}"

'''
Image: [H,W,C]
Mask: [Nr,H,W]
bbox: absolute value
'''
class WRandomTranslate(WTransform):
    def __init__(self,prob=0.6,pixels=60,image_fill_value=127,translate_horizontal=True,max_size=None):
        self.prob = prob
        self.pixels = pixels
        self.image_fill_value = image_fill_value
        self.translate_horizontal = translate_horizontal
        self.max_size = max_size
        self.pad_pixels = None

    def __call__(self, data_item):
        if not self.test_statu(WTransform.ABSOLUTE_COORDINATE):
            print(f"INFO: {self} better use absolute coordinate.")
        if not self.test_unstatu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need NHW format mask.")
        is_trans = tf.less_equal(tf.random_uniform(shape=()),self.prob)
        if self.max_size is not None:
            shape = tf.shape(data_item[IMAGE])
            if self.translate_horizontal:
                v = tf.minimum(self.pixels,self.max_size-shape[1])
            else:
                v = tf.minimum(self.pixels,self.max_size-shape[0])
            self.pad_pixels = tf.nn.relu(v)
        else:
            self.pad_pixels = self.pixels

        image = self.image_offset(data_item[IMAGE])
        self.cond_set(data_item,IMAGE,is_trans,image)
        if GT_MASKS in data_item:
            mask = self.mask_offset(data_item[GT_MASKS])
            self.cond_set(data_item,GT_MASKS,is_trans,mask)
        if GT_BOXES in data_item:
            bbox = self.box_offset(data_item[GT_BOXES])
            self.cond_set(data_item,GT_BOXES,is_trans,bbox)
        if GT_KEYPOINTS in data_item:
            keypoints = self.keypoints_offset(data_item[GT_KEYPOINTS])
            self.cond_set(data_item,GT_KEYPOINTS,is_trans,keypoints)
        return data_item

    def image_offset(self,image):
        if self.translate_horizontal:
            padding = [[0,0],[self.pad_pixels,0],[0,0]]
        else:
            padding = [[self.pad_pixels, 0], [0, 0], [0, 0]]
        return tf.pad(image,paddings=padding,constant_values=self.image_fill_value)

    def mask_offset(self,mask):
        if self.translate_horizontal:
            padding = [[0,0],[0,0],[self.pad_pixels,0]]
        else:
            padding = [[0, 0], [self.pad_pixels, 0], [0, 0]]
        return tf.pad(mask,paddings=padding)

    def box_offset(self,bbox):
        pad_pixels = tf.cast(self.pad_pixels,tf.float32)
        if self.translate_horizontal:
            offset = tf.convert_to_tensor([[0,pad_pixels,0,pad_pixels]],dtype=tf.float32)
        else:
            offset = tf.convert_to_tensor([[pad_pixels, 0, pad_pixels,0]], dtype=tf.float32)
        return bbox+offset

    def keypoints_offset(self,keypoints):
        pad_pixels = tf.cast(self.pad_pixels,tf.float32)
        if self.translate_horizontal:
            offset = tf.convert_to_tensor([[pad_pixels,0]],dtype=tf.float32)
        else:
            offset = tf.convert_to_tensor([[0, pad_pixels]], dtype=tf.float32)
        return keypoints+offset

    def __repr__(self):
        return f"{type(self).__name__}"

'''
image: in [0,255]
'''
class WColor(WTransform):
    def __init__(self,factor=1.18):
        self.factor = factor
    def __call__(self, data_item):
        image = color(data_item[IMAGE],factor=self.factor)
        data_item[IMAGE] = image
        return data_item

'''
mask: [Nr,H,W]
bboxes: absolute coordinate
'''
class WShear(WTransform):
    def __init__(self,shear_horizontal=False,level=0.06,replace=[128,128,128]):
        self.shear_horizontal = shear_horizontal
        self.level = level
        self.replace = replace

    def __call__(self, data_item):
        if not self.test_statu(WTransform.ABSOLUTE_COORDINATE):
            print(f"INFO: {self} better use absolute coordinate.")
        if not self.test_unstatu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need NHW format mask.")
        image = data_item[IMAGE]
        bboxes = data_item[GT_BOXES]
        mask = data_item.get(GT_MASKS,None)
        level = self.level
        replace = self.replace
        image,bboxes,mask = shear_with_bboxes(image, bboxes, mask, level, replace, self.shear_horizontal)
        data_item[IMAGE] = image
        data_item[GT_BOXES] = bboxes
        if GT_MASKS in data_item:
            data_item[GT_MASKS] = mask
        if GT_KEYPOINTS in data_item:
            print(f"WARNING: keypoints don't support transform {self}.")

        return data_item
    
    def __repr__(self):
        return f"{type(self).__name__}"

class RemoveMask(WTransform):
    def __init__(self):
        pass

    def __call__(self, data_item):
        res = data_item
        if GT_MASKS in res:
            del res[GT_MASKS]
        return res
'''
mask: [N,H,W]
'''
class RemoveSpecifiedInstance(WTransform):
    def __init__(self):
        pass

    def __call__(self, data_item):
        if not self.test_unstatu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need NHW format mask.")
        bboxes = data_item[GT_BOXES]
        labels = data_item[GT_LABELS]
        masks = data_item[GT_MASKS]
        image = data_item[IMAGE]
        
        remove = self.pred_fn(bboxes,labels,masks)
        keep = tf.logical_not(remove)
        
        if GT_BOXES in data_item:
            data_item[GT_BOXES] = tf.boolean_mask(data_item[GT_BOXES],keep)
        data_item[GT_LABELS] = tf.boolean_mask(data_item[GT_LABELS],keep)
        data_item[IMAGE] = self.remove_instance_in_image(image,masks,remove)
        if GT_MASKS in data_item:
            data_item[GT_MASKS]= tf.boolean_mask(masks, keep)
        if GT_KEYPOINTS in data_item:
            data_item[GT_KEYPOINTS]= tf.boolean_mask(data_item[GT_KEYPOINTS], keep)

        return data_item
    @staticmethod
    def remove_instance_in_image(image,masks,is_removeds,default_value=[127,127,127]):
        removed_image = tf.ones_like(image)*tf.convert_to_tensor([[default_value]],dtype=tf.uint8)
        masks = tf.expand_dims(masks,axis=-1)
        chl = channel(image)
        def fn(lhv,rhv):
            mask,is_removed = rhv
            select = tf.greater(mask,0)
            select = tf.tile(select,[1,1,chl])
            tmp_img = tf.where(select,removed_image,lhv)
            res = tf.cond(is_removed,lambda:tmp_img,lambda:lhv)
            return res
        return tf.foldl(fn,elems=(masks,is_removeds),back_prop=False,initializer=image)

    @staticmethod
    def pred_fn(bboxes,labels,masks):
        #an example
        return tf.equal(labels,1)
    
    def __repr__(self):
        return f"{type(self).__name__}"

class RandomNoise(WTransform):
    def __init__(self,probability=0.5,max_value=10.0):
        self.probabilitby = probability
        self.max_value = max_value
    
    def __call__(self,data_item):
        if data_item[IMAGE].dtype != tf.float32:
            data_item[IMAGE] = tf.cast(data_item[IMAGE], tf.float32)

        def fn():
            shape = tf.shape(data_item[IMAGE])
            return data_item[IMAGE]+tf.random_uniform(shape=shape,
                                                      minval=-self.max_value,
                                                      maxval=self.max_value)
            
        self.probability_fn_set(data_item,IMAGE,self.probabilitby,fn)
        
        return data_item

class ShowInfo(WTransform):
    def __init__(self,name="INFO"):
        self.name = name

    def __call__(self,data_item):
        tensors = [self.name,'img:',tf.shape(data_item[IMAGE])]
        if GT_BOXES in data_item:
            tensors += ['bboxes:',tf.shape(data_item[GT_BOXES])]
        if GT_LABELS in data_item:
            tensors += ['labels:',tf.shape(data_item[GT_LABELS])]
        if GT_MASKS in data_item:
            tensors += ['mask:',tf.shape(data_item[GT_MASKS])]
        if GT_KEYPOINTS in data_item:
            tensors += ['keypoints:',tf.shape(data_item[GT_KEYPOINTS])] +[data_item[GT_KEYPOINTS]]

        data_item[IMAGE] = tf.Print(data_item[IMAGE],tensors+[data_item[GT_LABELS]],summarize=1000)
        return data_item
'''
bbox: absolute coordinate
'''
class RandomMoveRect(WTransform):
    def __init__(self,probability=0.5,max_size=None):
        self.probability = probability
        self.max_size = max_size

    def __call__(self, data_item):
        if not self.test_statu(WTransform.ABSOLUTE_COORDINATE):
            print(f"INFO: {self} better use absolute coordinate.")
        is_proc = tf.less_equal(tf.random_uniform(shape=[]), self.probability)
        s_image = self.move_rect(data_item[IMAGE],data_item[GT_BOXES])
        image = tf.cond(is_proc,lambda:s_image,lambda:data_item[IMAGE])
        data_item[IMAGE] = image
        return data_item

    def npget_bboxes(self,env_bbox,image_size):
        x_l = env_bbox[1]
        y_t = env_bbox[0]
        x_r = image_size[1]-env_bbox[3]
        y_b = image_size[0]-env_bbox[2]
        mw = max(x_l,x_r)
        mh = max(y_t,y_b)

        H,W,_ = image_size

        if mw<20 or mh<20:
            return np.zeros([2,6],dtype=np.int32)

        if self.max_size is None:
            w = np.random.uniform(20,mw,size=()).astype(np.int32)
            h = np.random.uniform(20,mh,size=()).astype(np.int32)
        else:
            w = np.random.uniform(20,min(mw,self.max_size),size=()).astype(np.int32)
            h = np.random.uniform(20,min(mh,self.max_size),size=()).astype(np.int32)

        def get_box(w,h,index):
            index = index%4
            if index==0:
                if y_t<h:
                    return get_box(w,h,1)
                xmin = np.random.uniform(0,W-w-1,size=())
                ymin = np.random.uniform(0,y_t-h,size=())
                return np.array([ymin,xmin,ymin+h,xmin+w],dtype=np.int32)
            elif index==1:
                if x_r<w:
                    return get_box(w,h,2)
                xmin = np.random.uniform(W-x_r,W-w-1,size=())
                ymin = np.random.uniform(0,H-h-1,size=())
                return np.array([ymin,xmin,ymin+h,xmin+w],dtype=np.int32)
            elif index==2:
                if y_b<h:
                    return get_box(w,h,3)
                xmin = np.random.uniform(0,W-w-1,size=())
                ymin = np.random.uniform(H-y_b,H-h-1,size=())
                return np.array([ymin,xmin,ymin+h,xmin+w],dtype=np.int32)
            elif index==3:
                if x_l<w:
                    return get_box(w,h,0)
                xmin = np.random.uniform(0,x_l-w,size=())
                ymin = np.random.uniform(0,H-h-1,size=())
                return np.array([ymin,xmin,ymin+h,xmin+w],dtype=np.int32)
        index = np.random.uniform(0,3.1,size=(2)).astype(np.int32)
        _box0 = get_box(w,h,index[0])
        _box1 = get_box(w,h,index[1])
        box0 = np.array([_box0[0],_box0[1],0,_box0[2]-_box0[0],_box0[3]-_box0[1],3],dtype=np.int32)
        box1 = np.array([_box1[0],_box1[2],_box1[1],_box1[3],0,3],dtype=np.int32)
        return np.stack([box0,box1],axis=0)

    def move_rect(self,image,bboxes):
        env_bboxes = odb.tfbbox_of_boxes(bboxes)
        image_shape = btf.combined_static_and_dynamic_shape(image)
        boxes = tf.py_func(self.npget_bboxes,(env_bboxes,image_shape),tf.int32)
        need_move = tf.greater(boxes[0][3],5)
        src_begin = boxes[0][:3]
        src_size = boxes[0][3:]

        v = tf.slice(image,begin=src_begin,size=src_size)
        index = tf.reshape(boxes[1],[3,2])
        t_image = tfop.item_assign(tensor=image,v=v,index=index) 
        image = tf.cond(need_move,lambda:t_image,lambda:image)
        return image

    def __repr__(self):
        return f"{type(self).__name__}"


'''
mask: [N,H,W]
'''
class NPRemoveSpecifiedInstance(WTransform):
    def __init__(self,pred_fn=None):
        self.pred_fn = pred_fn

    def __call__(self, data_item):
        if not self.test_unstatu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need NHW format mask.")
        bboxes = data_item[GT_BOXES]
        labels = data_item[GT_LABELS]
        masks = data_item[GT_MASKS]
        image = data_item[IMAGE]

        remove = self.pred_fn(bboxes, labels, masks)
        keep = np.logical_not(remove)

        data_item[GT_BOXES] = data_item[GT_BOXES][keep]
        data_item[GT_LABELS] = data_item[GT_LABELS][keep]
        data_item[IMAGE] = self.remove_instance_in_image(image, masks, remove)
        if GT_MASKS in data_item:
            data_item[GT_MASKS] = masks[keep]
        if GT_KEYPOINTS:
            data_item[GT_KEYPOINTS] = data_item[GT_KEYPOINTS][keep]

        return data_item

    @staticmethod
    def remove_instance_in_image(image, masks, is_removeds, default_value=[127, 127, 127]):
        removed_image = np.ones_like(image) * np.array([[default_value]], dtype=np.uint8)
        masks = np.expand_dims(masks, axis=-1)
        chl = image.shape[-1]

        def fn(lhv, mask,is_removed):
            if is_removed:
                select = np.greater(mask, 0)
                select = np.tile(select, [1, 1, chl])
                tmp_img = np.where(select, removed_image, lhv)
            return tmp_img
        
        res_img = image
        for msk,is_rm in zip(masks,is_removeds):
            res_img = fn(res_img,msk,is_rm)
        
        return res_img


    @staticmethod
    def __pred_fn(bboxes, labels, masks):
        # an example
        return np.equal(labels, 1)

    def __repr__(self):
        return f"{type(self).__name__}"


class WTransformList(WTransform):
    def __init__(self,trans_list):
        self.trans_list = list(trans_list)
    def __call__(self, data_item):
        for trans in self.trans_list:
            data_item = trans(data_item)
        return data_item
    def __repr__(self):
        return f"{type(self).__name__}: "+str(self.trans_list)
    

'''
IMAGE:[H,W,C]
mask: [N,H,W]
bboxes: relative coordinate
'''
class AddFakeInstance(WTransform):
    def __init__(self):
        pass
    def __call__(self, data_item):
        if not self.test_unstatu(WTransform.ABSOLUTE_COORDINATE):
            print(f"WARNING: {self} need relative coordinate.")
        if not self.test_unstatu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need NHW format mask.")
        
        data_nr = tf.shape(data_item[GT_LABELS])[0]
        data_item = tf.cond(data_nr>0,lambda:data_item,partial(self.add_fake_obj,data_item))
        if GT_KEYPOINTS in data_item:
            print(f"WARNING: keypoints don't support transform {self}.")
        return data_item

    @btf.add_name_scope
    def add_fake_obj(self,data_item):
        img = data_item[IMAGE]
        H,W,C = btf.combined_static_and_dynamic_shape(img)
        bboxes = tf.constant([[0,0,1,1]],dtype=tf.float32)
        bboxes = tf.concat([data_item[GT_BOXES],bboxes],axis=0)
        label = tf.constant([0],dtype=data_item[GT_LABELS].dtype)
        labels = tf.concat([data_item[GT_LABELS],label],axis=0)
        
        if GT_MASKS in data_item:
            msk = data_item[GT_MASKS]
            mask = tf.ones([1,H,W],dtype=msk.dtype)
            mask = tf.concat([msk,mask],axis=0)
            
            data_item[GT_MASKS] = mask

        data_item[GT_BOXES] = bboxes
        data_item[GT_LABELS] = labels 

        return data_item
    
    def __repr__(self):
        return f"{type(self).__name__}: "+str(self.trans_list)

'''
IMAGE:[H,W,C]
mask: [N,H,W]
bboxes: relative coordinate
'''
class RemoveFakeInstance(WTransform):
    def __init__(self):
        pass
    
    def __call__(self, data_item):
        if not self.test_unstatu(WTransform.ABSOLUTE_COORDINATE):
            print(f"WARNING: {self} need relative coordinate.")
        if not self.test_unstatu(WTransform.HWN_MASK):
            print(f"WARNING: {self} need NHW format mask.")

        with tf.name_scope("remove_fake_instance"):
            if IS_CROWD in data_item:
                '''
                IS_CROWD的数据大于可能不与GT_LABELS等匹配，如果不删除会产生错误
                '''
                data_item.pop(IS_CROWD)
            mask = tf.greater(data_item[GT_LABELS],0)
            data_item = odt.boolean_mask_on_instances(data_item,mask,
                                                      labels_key=GT_LABELS,
                                                      length_key=GT_LENGTH)
        if GT_KEYPOINTS in data_item:
            print(f"WARNING: keypoints don't support transform {self}.")
        return data_item


class RemoveMask(WTransform):
    def __init__(self):
        pass

    def __call__(self, data_item):
        del data_item[GT_MASKS]
        return data_item
    
class GetSemanticMaskFromCOCO(WTransform):
    def __init__(self,num_classes,no_background=False):
        self.num_classes = num_classes
        self.no_background = no_background

    '''
    mask:[N,height,width]
    labels:[N]
    num_classes:() not include background 
    return:
    [num_classes,height,width]
    '''

    def sparse_mask_to_dense(self,mask, labels, num_classes, no_background=True):
        with tf.variable_scope("SparseMaskToDense"):
            if mask.dtype is not tf.bool:
                mask = tf.cast(mask, tf.bool)
            shape = btf.combined_static_and_dynamic_shape(mask)
            if no_background:
                out_shape = [num_classes, shape[1], shape[2]]
                labels = labels - 1
                init_res = tf.zeros(shape=out_shape, dtype=tf.bool)
            else:
                out_shape = [num_classes + 1, shape[1], shape[2]]
                init_res = tf.zeros(shape=out_shape, dtype=tf.bool)
                t_mask = tf.cast(mask,tf.int32)
                t_mask = tf.reduce_sum(t_mask,axis=0,keepdims=False)
                t_mask = tf.equal(t_mask,0)
                init_res = tfop.set_value(tensor=init_res, v=t_mask, index=tf.convert_to_tensor(0))

            def fn(merged_m, m, l):
                tmp_data = tfop.set_value(tensor=init_res, v=m, index=l)
                return tf.logical_or(merged_m, tmp_data)

            res = tf.foldl(lambda x, y: fn(x, y[0], y[1]), elems=(mask, labels), initializer=init_res, back_prop=False)
            res = tf.transpose(res,perm=[1,2,0])
            return res
        
    def __call__(self, data_item):
        data_item[GT_SEMANTIC_LABELS] = self.sparse_mask_to_dense(data_item[GT_MASKS],
                                                                  data_item[GT_LABELS],
                                                                  self.num_classes,
                                                                  self.no_background)
        return data_item
