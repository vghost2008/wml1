#coding=utf-8
import tensorflow as tf
import img_utils as wmli
import wtfop.wtfop_ops as wop
from collections import Iterable
import object_detection.bboxes as wml_bboxes
import time
from functools import partial
import wml_tfutils as wmlt
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

    def apply_to_images(self,func,data_item):
        return self.apply_with_filter(func,data_item,filter=self.is_image)

    def apply_to_masks(self,func,data_item):
        return self.apply_with_filter(func,data_item,filter=self.is_mask)

    def apply_to_images_and_masks(self,func,data_item):
        return self.apply_with_filter(func,data_item,
                                      filter=self.is_image_or_mask)

    def apply_to_bbox(self,func,data_item):
        return self.apply_with_filter(func,data_item,filter=self.is_bbox)

    def apply_to_label(self,func,data_item):
        return self.apply_with_filter(func,data_item,filter=self.is_label)

    def apply_with_filter(self,func,data_item,filter):
        res = {}
        for k,v in data_item.items():
            if filter(k):
                res[k] = func(v)
            else:
                res[k] = v
        return res
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

'''
mask:[N,H,W]
'''
def distorted_bounding_box_and_mask_crop(image,
                                mask,
                                labels,
                                bboxes,
                                min_object_covered=0.3,
                                aspect_ratio_range=(0.9, 1.1),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                filter_threshold=0.3,
                                scope=None):
    dst_image, labels, bboxes, bbox_begin,bbox_size,bboxes_mask= \
            distorted_bounding_box_crop(image, labels, bboxes,
                                        area_range=area_range,
                                        min_object_covered=min_object_covered,
                                        aspect_ratio_range=aspect_ratio_range,
                                        max_attempts=max_attempts,
                                        filter_threshold=filter_threshold,
                                        scope=scope)
    mask = tf.boolean_mask(mask,bboxes_mask)
    mask = tf.transpose(mask,perm=[1,2,0])
    mask = tf.slice(mask,bbox_begin,bbox_size)
    dst_mask = tf.transpose(mask,perm=[2,0,1])
    return dst_image,dst_mask,labels, bboxes, bbox_begin,bbox_size,bboxes_mask

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


class ResizeShortestEdge(WTransform):
    def __init__(self,short_edge_length=range(640,801,32),align=1,resize_method=tf.image.ResizeMethod.BILINEAR,seed=None):
        if isinstance(short_edge_length, Iterable):
            short_edge_length = list(short_edge_length)
        elif not isinstance(short_edge_length, list):
            short_edge_length = [short_edge_length]
        self.short_edge_length = short_edge_length
        self.align = align
        self.resize_method = resize_method
        self.seed = seed

    def __call__(self, data_item):
        with tf.name_scope("resize_shortest_edge"):
            if len(self.short_edge_length) == 1:
                func = partial(resize_img,limit=[self.short_edge_length[0],-1],align=self.align,
                                  resize_method=self.resize_method)

            else:
                idx = tf.random_uniform(shape=(),
                                        minval=0,maxval=len(self.short_edge_length),
                                        dtype=tf.int32,
                                        seed=self.seed)
                s = tf.gather(self.short_edge_length,idx)
                func = partial(resize_img,limit=[s,-1],align=self.align,resize_method=self.resize_method)
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
            func = partial(wml_bboxes.tfabsolutely_boxes_to_relative_boxes,width=size[1],height=size[0])
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

class AddBoxLens(WTransform):
    def __init__(self,box_key="gt_boxes",gt_len_key="gt_length"):
        self.box_key = box_key
        self.gt_len_key = gt_len_key

    def __call__(self, data_item):
        gt_len = tf.shape(data_item[self.box_key])[0]
        data_item[self.gt_len_key] = gt_len
        return data_item

