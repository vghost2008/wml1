#coding=utf-8
from itertools import count
import tensorflow as tf
import img_utils as wmli
import wtfop.wtfop_ops as wop
from collections import Iterable
import object_detection2.bboxes as wml_bboxes
import time
from functools import partial
import basic_tftools as btf
from object_detection2.standard_names import *
import numpy as np
from .autoaugment import *
from thirdparty.aug.autoaugment import distort_image_with_autoaugment

'''
所有的变换都只针对一张图, 部分可以兼容同时处理一个batch
'''
class WTransform(object):
    def is_image(self,key):
        return ('image' in key) or ('img' in key)

    def is_mask(self,key):
        return ('mask' in key)

    def is_image_or_mask(self,key):
        return ('image' in key) or ('img' in key) or ('mask' in key)

    def is_bbox(self,key):
        return ('box' in key)

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
    def __str__(self):
        return type(self).__name__

    @staticmethod
    def select(pred,true_v,false_v):
        return tf.cond(pred,lambda:true_v,lambda:false_v)

    @staticmethod
    def cond_set(dict_data,key,pred,v):
        dict_data[key] = tf.cond(pred,lambda:v,lambda:dict_data[key])

    @staticmethod
    def probability_set(dict_data,key,prob,v):
        pred = tf.less_equal(tf.random_uniform(shape=()),prob)
        dict_data[key] = tf.cond(pred,lambda:v,lambda:dict_data[key])

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
limit: 参考wop.get_image_resize_size
align: 参考wop.get_image_resize_size
'''
def resize_img(img,limit,align,resize_method=tf.image.ResizeMethod.BILINEAR):
    with tf.name_scope("resize_img"):
        new_size = wop.get_image_resize_size(size=tf.shape(img)[0:2], limit=limit, align=align)
        return tf.image.resize_images(img, new_size,method=resize_method)

def distort_color(image, color_ordering=6, fast_mode=False,
            b_max_delta=0.1,
            c_lower = 0.8,
            c_upper = 1.2,
            s_lower = 0.5,
            s_upper = 1.5,
            h_max_delta = 0.1,
            scope=None,seed=None):
    with tf.name_scope(scope, 'distort_color', [image]):
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
            elif color_ordering == 7:
                return image
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        return image

def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered=0.3,
                                aspect_ratio_range=(0.9, 1.1),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                filter_threshold=0.3,
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
                use_image_if_no_bounding_boxes=True,
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
        bboxes = wml_bboxes.bboxes_resize(distort_bbox, bboxes)
        '''
        仅保留交叉面积大于threshold的bbox
        '''
        labels, bboxes,mask = wml_bboxes.bboxes_filter_overlap(labels, bboxes,
                                                   threshold=filter_threshold,
                                                   assign_negative=False)
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
    def __init__(self):
        pass
    def __call__(self, data_item):
        is_flip = tf.greater(tf.random_uniform(shape=[]),0.5)
        func = tf.image.flip_left_right
        data_item = self.apply_to_images_and_masks(func,data_item,runtime_filter=is_flip)
        func2 = wml_bboxes.bboxes_flip_left_right
        data_item = self.apply_to_bbox(func2,data_item,runtime_filter=is_flip)
        return data_item

'''
Boxes: relative coordinate
mask:H,W,N format
'''
class RandomFlipUpDown(WTransform):
    def __init__(self):
        pass
    def __call__(self, data_item):
        is_flip = tf.greater(tf.random_uniform(shape=[]),0.5)
        func = tf.image.flip_up_down
        data_item = self.apply_to_images_and_masks(func,data_item,runtime_filter=is_flip)
        func2 = wml_bboxes.bboxes_flip_up_down
        data_item = self.apply_to_bbox(func2,data_item,runtime_filter=is_flip)
        return data_item

class AutoAugment(WTransform):
    def __init__(self,augmentation_name="v0"):
        self.augmentation_name = augmentation_name
    def __call__(self, data_item:dict):
        if GT_MASKS in data_item:
            data_item.pop(GT_MASKS)
        image,bbox = distort_image_with_autoaugment(data_item[IMAGE],data_item[GT_BOXES],self.augmentation_name)
        data_item[IMAGE] = image
        data_item[GT_BOXES] = bbox

        return data_item
'''
Boxes: relative coordinate
mask:H,W,N format
'''
class RandomRotate(WTransform):
    def __init__(self,clockwise=True):
        self.clockwise = clockwise
        if clockwise:
            self.k = 1
        else:
            self.k = 3

    def __call__(self, data_item):
        is_rotate = tf.greater(tf.random_uniform(shape=[]),0.5)
        func = partial(tf.image.rot90,k=self.k)
        data_item = self.apply_to_images_and_masks(func,data_item,runtime_filter=is_rotate)
        func2 = partial(wml_bboxes.bboxes_rot90,clockwise=self.clockwise)
        data_item = self.apply_to_bbox(func2,data_item,runtime_filter=is_rotate)
        return data_item

'''
Mask: [Nr,H,W]
bboxes: absolute coordinate
'''
class RandomRotateAnyAngle(WTransform):
    def __init__(self,max_angle=60,rotate_probability=0.5,enable=True):
        self.max_angle = max_angle
        self.rotate_probability = rotate_probability
        self.enable = enable

    def __call__(self, data_item):
        if not self.enable:
            return data_item
        is_rotate = tf.less(tf.random_uniform(shape=[]),self.rotate_probability)
        if self.max_angle is not None:
            angle = tf.random_uniform(shape=(),minval=-self.max_angle,maxval=self.max_angle)
        else:
            angle = tf.random_uniform(shape=(), minval=-180,maxval=180)

        r_image = wop.tensor_rotate(image=data_item[IMAGE],angle=angle)
        WTransform.cond_set(data_item, IMAGE, is_rotate, r_image)

        if GT_MASKS in data_item:
            r_mask,r_bboxes = wop.mask_rotate(image=data_item[GT_MASKS],angle=angle)
            WTransform.cond_set(data_item,GT_MASKS,is_rotate,r_mask)
            WTransform.cond_set(data_item, GT_BOXES, is_rotate, r_bboxes)

        return data_item
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
    def __str__(self):
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

class MaskNHW2HWN(WTransform):
    def __init__(self):
        pass

    def __call__(self, data_item):
        with tf.name_scope("MaskNHW2HWN"):
            func = partial(tf.transpose,perm=[1,2,0])
            return self.apply_to_masks(func,data_item)

class MaskHWN2NHW(WTransform):
    def __init__(self):
        pass

    def __call__(self, data_item):
        with tf.name_scope("MaskHWN2NHW"):
            func = partial(tf.transpose,perm=[2,0,1])
            return self.apply_to_masks(func,data_item)

class DelHeightWidth(WTransform):
    def __call__(self, data_item):
        if 'height' in data_item:
            del data_item['height']
        if 'width' in data_item:
            del data_item['width']
        return data_item

class UpdateHeightWidth(WTransform):
    def __call__(self, data_item):
        shape = tf.shape(data_item[IMAGE])
        if HEIGHT in data_item:
            data_item[HEIGHT] = shape[0]
        if WIDTH in data_item:
            data_item[WIDTH] = shape[1]
        return data_item

class AddSize(WTransform):
    def __init__(self,img_key='image'):
        self.img_key = img_key
    def __call__(self, data_item):
        data_item['size'] = tf.shape(data_item[self.img_key])[0:2]
        return data_item

class BBoxesRelativeToAbsolute(WTransform):
    def __init__(self,img_key='image'):
        self.img_key = img_key

    def __call__(self, data_item):
        with tf.name_scope("BBoxesRelativeToAbsolute"):
            size = tf.shape(data_item[self.img_key])
            func = partial(wml_bboxes.tfrelative_boxes_to_absolutely_boxes,width=size[1],height=size[0])
            return self.apply_to_bbox(func,data_item)

'''
prossed batched data
'''
class BBoxesAbsoluteToRelative(WTransform):
    def __init__(self,img_key='image'):
        self.img_key = img_key

    def __call__(self, data_item):
        with tf.name_scope("BBoxesRelativeToAbsolute"):
            size = tf.shape(data_item[self.img_key])[1:3]
            func = partial(wml_bboxes.tfbatch_absolutely_boxes_to_relative_boxes,width=size[1],height=size[0])
            return self.apply_to_bbox(func,data_item)

'''
如果有Mask分支，Mask必须先转换为HWN的模式
'''
class ResizeToFixedSize(WTransform):
    def __init__(self,size=[224,224],resize_method=tf.image.ResizeMethod.BILINEAR,channels=3):
        self.resize_method = resize_method
        self.size = size
        self.channels = channels

    def __call__(self, data_item):
        with tf.name_scope("resize_image"):
            func = partial(tf.image.resize_images,size=self.size, method=self.resize_method)
            #data_item['gt_boxes'] = tf.Print(data_item['gt_boxes'],[tf.shape(data_item['gt_boxes']),data_item['fileindex']])
            items = self.apply_to_images_and_masks(func,data_item)
            if self.channels is not None:
                func = partial(tf.reshape,shape=self.size+[self.channels])
                items = self.apply_to_images(func,items)
            return items

class NoTransform(WTransform):
    def __init__(self,id=None):
        self.id = id
    def __call__(self, data_item):
        if self.id is None:
            return data_item
        else:
            data_item[IMAGE] = tf.Print(data_item[IMAGE],["id:",self.id])
            return data_item

class RandomSelectSubTransform(WTransform):
    def __init__(self,trans,probs=None):
        self.trans = [WTransformList(v) if isinstance(v,Iterable) else v for v in trans]
        if probs is None:
            self.probs = [1.0/len(trans)]*len(trans)
        else:
            self.probs = probs

    def __call__(self, data_item:dict):
        all_items = []
        for t in self.trans:
            all_items.append(t(data_item))
        res_data_items = {}
        index = tf.random_uniform(shape=(),maxval=len(self.trans),dtype=tf.int32)
        for k in data_item.keys():
            datas = []
            for i,v in enumerate(all_items):
                datas.append(v[k])
            res_data_items[k] = tf.stack(datas,axis=0)[index]
        return res_data_items

class AddBoxLens(WTransform):
    def __init__(self,box_key="gt_boxes",gt_len_key="gt_length"):
        self.box_key = box_key
        self.gt_len_key = gt_len_key

    def __call__(self, data_item):
        gt_len = tf.shape(data_item[self.box_key])[0]
        data_item[self.gt_len_key] = gt_len
        return data_item

class WDistortColor(WTransform):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
    def __call__(self, data_item):
        func = partial(distort_color,**self.kwargs)
        return self.apply_to_images(func,data_item)

class WRandomDistortColor(WTransform):
    def __init__(self,probability=0.5,**kwargs):
        self.kwargs = kwargs
        self.probability = probability
    def __call__(self, data_item):
        is_dis = tf.less_equal(tf.random_uniform(shape=[]),self.probability)
        func = partial(distort_color,**self.kwargs)
        return self.apply_to_images(func,data_item,runtime_filter=is_dis)

class WRandomBlur(WTransform):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
    def __call__(self, data_item):
        return self.apply_to_images(random_blur,data_item)

class WTransImgToFloat(WTransform):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
    def __call__(self, data_item):
        func = partial(tf.cast, dtype=tf.float32)
        return self.apply_to_images(func,data_item)

class WTransImgToGray(WTransform):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
    def __call__(self, data_item):
        return self.apply_to_images(tf.image.rgb_to_grayscale,data_item)

class WPerImgStandardization(WTransform):
    def __init__(self,**kwargs):
        self.kwargs = kwargs

    def __call__(self, data_item):
        return self.apply_to_images(tf.image.per_image_standardization,data_item)
'''
如果有Mask分支，Mask必须先转换为HWN的模式
'''
class SampleDistortedBoundingBox(WTransform):
    def __init__(self,
                 min_object_covered=0.3,
                 aspect_ratio_range=(0.9, 1.1),
                 area_range=(0.1, 1.0),
                 max_attempts=100,
                 filter_threshold=0.3,
                 ):
        self.min_object_covered = min_object_covered
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts
        self.filter_threshold = filter_threshold

    def __call__(self, data_item):
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
                                        scope="distorted_bounding_box_crop")
        data_item[IMAGE] = dst_image
        data_item[GT_LABELS] = labels
        data_item[GT_BOXES] = bboxes

        if GT_MASKS in data_item:
            def func(mask):
                mask = tf.boolean_mask(mask,bboxes_mask)
                mask = tf.slice(mask,bbox_begin,bbox_size)
                return mask
            data_item = self.apply_to_masks(func,data_item)

        return data_item

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
            mask = tf.boolean_mask(mask,bboxes_mask)
            mask = tf.slice(mask,bbox_begin,bbox_size)
            self.cond_set(data_item, GT_MASKS, distored, mask)

        return data_item

class WTransLabels(WTransform):
    def __init__(self,id_to_label):
        '''
        原有的label称为id, 变换后的称为label
        id_to_label为一个字典, key为id, value为label
        '''
        self.id_to_label = id_to_label

    def __call__(self, data_item):
        labels = data_item[GT_LABELS]
        labels = wop.int_hash(labels,self.id_to_label)
        data_item[GT_LABELS] = labels
        return data_item

'''
Mask必须为NHW格式
'''
class WRemoveCrowdInstance(WTransform):
    def __init__(self,remove_crowd=True):
        self.remove_crowd = remove_crowd

    def __call__(self, data_item):

        if (IS_CROWD not in data_item) or not self.remove_crowd:
            return data_item

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
        return data_item
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

'''
image: [B,H,W,C]
如果有Mask分支，Mask必须为[B,N,H,W]
'''
class PadtoAlign(WTransform):
    def __init__(self,align=1):
        self.align = align

    def __call__(self, data_item):
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

'''
用于将多个图像拼在一起，生成一个包含更多（可能也更清晰）的小目标
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
        B,H,W,C = btf.combined_static_and_dynamic_shape(data_item[IMAGE])
        if B<2:
            return data_item

        indexs = tf.range((B//2)*2)
        indexs = tf.random_shuffle(indexs,seed=time.time())
        indexs = tf.reshape(indexs,[B//2,2])
        if self.probability is None:
            nr = tf.minimum(B//2,int(self.nr+0.5))
        else:
            p = tf.less(tf.random_uniform(shape=()),self.probability)
            nr = tf.cond(p,lambda:1,lambda:0)
        need_trans = tf.greater(nr,0)
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
        pad_value = max_length-tf.shape(data_item[GT_LABELS])[1]
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

        def cond_set_value(data_dict,key,pred,value,indexs):
            org_v = data_dict[key]
            shape = btf.combined_static_and_dynamic_shape(org_v)
            cond_v = tf.scatter_nd(tf.reshape(indexs,[-1,1]),value,shape)
            is_set = tf.cast(btf.indices_to_dense_vector(indexs,size=shape[0]),tf.bool)
            cond_v = tf.where(is_set,cond_v,org_v)
            WTransform.cond_set(data_dict,key,pred,cond_v)
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
            cond_set_value(data_item,GT_MASKS,need_trans,mask,o_indexs)
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
        cond_set_value(data_item, IMAGE, need_trans, image,o_indexs)
        cond_set_value(data_item, GT_BOXES, need_trans, bboxes,o_indexs)
        cond_set_value(data_item, GT_LABELS, need_trans, labels,o_indexs)
        cond_set_value(data_item, GT_LENGTH, need_trans, length,o_indexs)

        return data_item

    @staticmethod
    def fitto(width,height,target_w,target_h):
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
            if mask is not None:
                m = Stitch.concat_mask([mask[id0][:,:,:len0],mask[id1][:,:,:len1]],widths,heights,W,H,ih)
                m = np.pad(m,[(0,0),(0,0),(0,max_length-len0-len1)],'constant', constant_values=(0,0))
                mask[id0] = m
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

class WRandomEqualize(WTransform):
    def __init__(self,prob=0.8):
        self.prob = prob

    def __call__(self, data_item):
        image = equalize(data_item[IMAGE])
        self.probability_set(data_item,IMAGE,self.prob,image)
        return data_item

class WRandomCutout(WTransform):
    def __init__(self,pad_size=8,prob=0.8):
        self.prob = prob
        self.pad_size = pad_size

    def __call__(self, data_item):
        image = cutout(data_item[IMAGE],pad_size=self.pad_size)
        self.probability_set(data_item,IMAGE,self.prob,image)
        return data_item

'''
Image: [H,W,C]
Mask: [Nr,H,W]
bbox: absolute value
'''
class WRandomTranslateX(WTransform):
    def __init__(self,prob=0.6,pixels=60,image_fill_value=127):
        self.prob = prob
        self.pixels = pixels
        self.image_fill_value = image_fill_value
    def __call__(self, data_item):
        is_trans = tf.less_equal(tf.random_uniform(shape=()),self.prob)
        image = self.image_offset(data_item[IMAGE])
        self.cond_set(data_item,IMAGE,is_trans,image)
        if GT_MASKS in data_item:
            mask = self.mask_offset(data_item[GT_MASKS])
            self.cond_set(data_item,GT_MASKS,is_trans,mask)
        if GT_BOXES in data_item:
            bbox = self.box_offset(data_item[GT_BOXES])
            self.cond_set(data_item,GT_BOXES,is_trans,bbox)
        return data_item

    def image_offset(self,image):
        padding = [[0,0],[self.pixels,0],[0,0]]
        return tf.pad(image,paddings=padding,constant_values=self.image_fill_value)

    def mask_offset(self,mask):
        padding = [[0,0],[0,0],[self.pixels,0]]
        return tf.pad(mask,paddings=padding)

    def box_offset(self,bbox):
        offset = tf.convert_to_tensor([[0,self.pixels,0,self.pixels]],dtype=tf.float32)
        return bbox+offset

class WColor(WTransform):
    def __init__(self,factor=1.18):
        self.factor = factor
    def __call__(self, data_item):
        image = color(data_item[IMAGE],factor=self.factor)
        data_item[IMAGE] = image
        return data_item

class WTransformList(WTransform):
    def __init__(self,trans_list):
        self.trans_list = list(trans_list)
    def __call__(self, data_item):
        for trans in self.trans_list:
            data_item = trans(data_item)
        return data_item
    def __str__(self):
        return "WTransformList: "+str(self.trans_list)

