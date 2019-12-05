#coding=utf-8
#import pydicom as dcm
import scipy.misc
import matplotlib.image as mpimg
import numpy as np
import wml_utils as wmlu
import shutil
import os
import cv2
import tensorflow as tf
import copy
import random
import itertools
import functools
import wml_tfutils as wmlt
import time

'''def dcm_to_jpeg(input_file,output_file):
    ds = dcm.read_file(input_file)
    pix = ds.pixel_array
    scipy.misc.imsave(output_file, pix)
    return pix.shape'''

def normal_image(image,min=0,max=255,dtype=np.uint8):
    if not isinstance(image,np.ndarray):
        image = np.array(image)
    t = image.dtype
    if t!=np.float32:
        image = image.astype(np.float32)
    i_min = image.min()
    i_max = image.max()
    image = (image-float(i_min))*float(max-min)/float(i_max-i_min)+float(min)

    if dtype!=np.float32:
        image = image.astype(dtype)

    return image


'''def dcms_to_jpegs(input_dir,output_dir):
    input_files = wmlu.recurse_get_filepath_in_dir(input_dir,suffix=".dcm")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for file in input_files:
        print('trans file \"%s\"'%(os.path.basename(file)))
        output_file = os.path.join(output_dir,wmlu.base_name(file)+".png")
        dcm_to_jpeg(file,output_file)'''

def blur(img,size=(5,5),sigmaX=0,sigmaY=0):
    #old_shape = img.get_shape()
    old_shape = tf.shape(img)
    def func(img):
        dst = np.zeros_like(img)
        cv2.GaussianBlur(img,dst=dst,ksize=size,sigmaX=sigmaX,sigmaY=sigmaY)
        return dst
    res = tf.py_func(func,[img],Tout=[img.dtype])[0]
    res = tf.reshape(res,old_shape)
    return res

def random_blur(img,size=(5,5),sigmaX=0,sigmaY=0,prob=0.5):
    return tf.cond(tf.greater(tf.random_uniform(shape=[]), prob),
                   lambda: (img),
                   lambda: blur(img,size,sigmaX,sigmaY))

def to_jpeg(input_file,output_file):
    _input_file = input_file.lower()
    if _input_file.endswith(".jpg") or _input_file.endswith(".jpeg"):
        shutil.copyfile(input_file,output_file)
        #return None
    else:
        pix = mpimg.imread(input_file)
        scipy.misc.imsave(output_file, pix)
        #return pix.shape

def npgray_to_rgb(img):
    if img.ndim == 2:
        img = np.expand_dims(img,axis=2)
    shape = img.shape
    if shape[2] == 1:
        img = np.concatenate([img, img, img], axis=2)
    return img

def adjust_image_value_range(img):
    min = np.min(img)
    max = np.max(img)
    img = (img-min)*255.0/(max-min)
    return img

def npgray_to_rgbv2(img):
    img = adjust_image_value_range(img)
    def r(v):
        return np.where(v >= 127., 255., v * 255. / 127)
    def g(v):
        return (1. - np.abs(v - 127.) / 127.) * 255.
    def b(v):
        return np.where(v<=127.,255.,(1.-(v-127.)/127)*255.)
    if img.ndim == 2:
        img = np.expand_dims(img,axis=2)
    shape = img.shape
    if shape[2] == 1:
        img = np.concatenate([r(img), g(img), b(img)], axis=2)
    return img

def nprgb_to_gray(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    img_gray = R * 299. / 1000 + G * 587. / 1000 + B * 114. / 1000
    return img_gray

def npbatch_rgb_to_gray(img):
    R = img[:,:, :, 0]
    G = img[:,:, :, 1]
    B = img[:,:, :, 2]
    img_gray = R * 299. / 1000 + G * 587. / 1000 + B * 114. / 1000
    return img_gray

'''
images:[batch_size,h,w,3] or [h,w,3]
'''
def rgb_to_grayscale(images,keep_channels=True):
    images = tf.image.rgb_to_grayscale(images)
    if keep_channels:
        last_dim = images.get_shape().ndims-1
        images = tf.concat([images,images,images],axis=last_dim)

    return images

def merge_image(src,dst,alpha):
    src = adjust_image_value_range(src)
    dst = adjust_image_value_range(dst)
    if len(dst.shape)<3:
        dst = np.expand_dims(dst,axis=2)
    if src.shape[2] != dst.shape[2]:
        if src.shape[2] == 1:
            src = npgray_to_rgb(src)
        if dst.shape[2] == 1:
            dst = npgray_to_rgb(dst)

    return src*(1.0-alpha)+dst*alpha

def merge_hotgraph_image(src,dst,alpha):
    if len(dst.shape)<3:
        dst = np.expand_dims(dst,axis=2)
    if src.shape[2] != dst.shape[2]:
        if src.shape[2] == 1:
            src = npgray_to_rgb(src)

    src = adjust_image_value_range(src)/255.
    dst = adjust_image_value_range(dst)/255.
    mean = np.mean(dst)
    rgb_dst = npgray_to_rgbv2(dst)/255.

    return np.where(dst>mean,src*(1.0-(2.*dst-1.)*alpha)+rgb_dst*(2.*dst-1.)*alpha,src)

'''def resize_img(img,size):

    image_shape = img.shape

    if size[0]==image_shape[0] and size[1]==image_shape[1]:
        return img

    h_scale = (float(size[0])+0.45)/float(image_shape[0])
    w_scale = (float(size[1])+0.45)/float(image_shape[1])
    if len(img.shape)==2:
        return scipy.ndimage.zoom(img, [h_scale, w_scale])
    else:
        return scipy.ndimage.zoom(img, [h_scale, w_scale,1])'''
def resize_img(img,size):

    image_shape = img.shape

    if size[0]==image_shape[0] and size[1]==image_shape[1]:
        return img
    return cv2.resize(img,dsize=size)

def flip_left_right_images(images):
    return tf.map_fn(tf.image.flip_left_right,elems=images,back_prop=False)

def flip_up_down_images(images):
    return tf.map_fn(tf.image.flip_up_down,elems=images,back_prop=False)

def random_flip_left_right_images(images,flip_probs=0.5):
    v = tf.random_uniform(shape=(),minval=0.0,maxval=1.0,dtype=tf.float32)
    cond = tf.greater(v,flip_probs)
    return tf.cond(cond,lambda:images,lambda:flip_left_right_images(images))

def random_flip_up_down_images(images,flip_probs=0.5):
    v = tf.random_uniform(shape=(),minval=0.0,maxval=1.0,dtype=tf.float32)
    cond = tf.greater(v,flip_probs)
    return tf.cond(cond,lambda:images,lambda:flip_up_down_images(images))

def random_flip_left_right_images_array(images,flip_probs=0.5):
    v = tf.random_uniform(shape=(),minval=0.0,maxval=1.0,dtype=tf.float32)
    cond = tf.greater(v,flip_probs)
    return map(lambda image:tf.cond(cond,lambda:image,lambda:flip_left_right_images(image)),images)

def random_flip_up_down_images_array(images,flip_probs=0.5):
    v = tf.random_uniform(shape=(),minval=0.0,maxval=1.0,dtype=tf.float32)
    cond = tf.greater(v,flip_probs)
    return map(lambda image:tf.cond(cond,lambda:image,lambda:flip_up_down_images(image)),images)

'''
box:ymin,xmin,ymax,xmax, relative corrdinate
'''
def crop_img(img,box):
    shape = img.shape
    box = np.array(box)
    box = np.minimum(box,1.0)
    box = np.maximum(box,0.0)
    ymin = int((shape[0]-1)*box[0]+0.5)
    ymax = int((shape[0]-1)*box[2]+1+0.5)
    xmin = int((shape[1]-1)*box[1]+0.5)
    xmax = int((shape[1]-1)*box[3]+1+0.5)
    if len(shape)==2:
        return img[ymin:ymax,xmin:xmax]
    else:
        return img[ymin:ymax,xmin:xmax,:]

'''
box:[ymin,xmin,ymax,xmax], relative coordinate
crop_size:[heigh,width] absolute pixel size.
'''
def crop_and_resize(img,box,crop_size):
    img = crop_img(img,box)
    return resize_img(img,crop_size)

'''
box:[N,4] ymin,xmin,ymax,xmax, relative corrdinate
'''
def crop_and_resize_imgs(img,boxes,crop_size):
    res_imgs = []
    for box in boxes:
        sub_img = crop_and_resize(img,box,crop_size)
        res_imgs.append(sub_img)

    return np.stack(res_imgs,axis=0)

'''
对图像image进行剪切，生成四个角及中间五个不同位置的图像, 生成的图像大小为[height,width]
如果resize_size不为None， 那么生成的图像会被缩放为resize_size指定的大小
'''
def crop_image(image,width,height,resize_size=None):
    shape = tf.shape(image)
    images = []
    img = tf.image.crop_to_bounding_box(image,0,0,height,width)
    images.append(img)
    img = tf.image.crop_to_bounding_box(image, 0, shape[1] - width, height, width)
    images.append(img)
    img = tf.image.crop_to_bounding_box(image, shape[0] - height, 0, height, width)
    images.append(img)
    img = tf.image.crop_to_bounding_box(image,shape[0]-height,shape[1]-width,height,width)
    images.append(img)
    img = tf.image.crop_to_bounding_box(image, (shape[0] - height)//2, (shape[1] - width)//2, height, width)
    images.append(img)

    if resize_size is not None:
        images = tf.stack(images,axis=0)
        return tf.image.resize_images(images,resize_size)
    else:
        return tf.stack(images,axis=0)

def concat_images(images):
    new_images = []
    size = tf.shape(images[0])[1:3]
    for img in images:
        new_images.append(tf.image.resize_bilinear(img,size=size))
    return tf.concat(new_images,axis=2)
'''
img:[H,W]/[H,W,C]
rect:[ymin,xmin,ymax,xmax]
'''
def sub_image(img,rect):
    return copy.deepcopy(img[rect[0]:rect[2],rect[1]:rect[3]])

def nprandom_crop(img,size):
    size = list(copy.deepcopy(size))
    x_begin = 0
    y_begin = 0
    if img.shape[0]>size[0]:
        y_begin = random.randint(0,img.shape[0]-size[0])
    else:
        size[0] = img.shape[0]
    if img.shape[1]>size[1]:
        x_begin = random.randint(0,img.shape[1]-size[1])
    else:
        size[1] = img.shape[1]

    rect = [y_begin,x_begin,y_begin+size[0],x_begin+size[1]]
    return sub_image(img,rect)

def imread(filepath):
    img = cv2.imread(filepath)
    cv2.cvtColor(img,cv2.COLOR_BGR2RGB,img)
    return img

def imsave(filename,img):
    imwrite(filename,img)

def imwrite(filename, img):
    if len(img.shape)==3 and img.shape[2]==3:
        img = copy.deepcopy(img)
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR,img)
    cv2.imwrite(filename, img)

def resize_to_range(image,
                    masks=None,
                    min_dimension=None,
                    max_dimension=None,
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=False,
                    pad_to_max_dimension=False,
                    per_channel_pad_value=(0, 0, 0)):
  """Resizes an image so its dimensions are within the provided value.

  The output size can be described by two cases:
  1. If the image can be rescaled so its minimum dimension is equal to the
     provided value without the other dimension exceeding max_dimension,
     then do so.
  2. Otherwise, resize so the largest dimension is equal to max_dimension.

  Args:
    image: A 3D tensor of shape [height, width, channels]
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks.
    min_dimension: (optional) (scalar) desired size of the smaller image
                   dimension.
    max_dimension: (optional) (scalar) maximum allowed size
                   of the larger image dimension.
    method: (optional) interpolation method used in resizing. Defaults to
            BILINEAR.
    align_corners: bool. If true, exactly align all 4 corners of the input
                   and output. Defaults to False.
    pad_to_max_dimension: Whether to resize the image and pad it with zeros
      so the resulting image is of the spatial size
      [max_dimension, max_dimension]. If masks are included they are padded
      similarly.
    per_channel_pad_value: A tuple of per-channel scalar value to use for
      padding. By default pads zeros.

  Returns:
    Note that the position of the resized_image_shape changes based on whether
    masks are present.
    resized_image: A 3D tensor of shape [new_height, new_width, channels],
      where the image has been resized (with bilinear interpolation) so that
      min(new_height, new_width) == min_dimension or
      max(new_height, new_width) == max_dimension.
    resized_masks: If masks is not None, also outputs masks. A 3D tensor of
      shape [num_instances, new_height, new_width].
    resized_image_shape: A 1D tensor of shape [3] containing shape of the
      resized image.

  Raises:
    ValueError: if the image is not a 3D tensor.
  """
  if len(image.get_shape()) != 3:
    raise ValueError('Image should be 3D tensor')

  def _resize_landscape_image(image):
    # resize a landscape image
    return tf.image.resize_images(
        image, tf.stack([min_dimension, max_dimension]), method=method,
        align_corners=align_corners, preserve_aspect_ratio=True)

  def _resize_portrait_image(image):
    # resize a portrait image
    return tf.image.resize_images(
        image, tf.stack([max_dimension, min_dimension]), method=method,
        align_corners=align_corners, preserve_aspect_ratio=True)

  with tf.name_scope('ResizeToRange', values=[image, min_dimension]):
    if image.get_shape().is_fully_defined():
      if image.get_shape()[0] < image.get_shape()[1]:
        new_image = _resize_landscape_image(image)
      else:
        new_image = _resize_portrait_image(image)
      new_size = tf.constant(new_image.get_shape().as_list())
    else:
      new_image = tf.cond(
          tf.less(tf.shape(image)[0], tf.shape(image)[1]),
          lambda: _resize_landscape_image(image),
          lambda: _resize_portrait_image(image))
      new_size = tf.shape(new_image)

    if pad_to_max_dimension:
      channels = tf.unstack(new_image, axis=2)
      if len(channels) != len(per_channel_pad_value):
        raise ValueError('Number of channels must be equal to the length of '
                         'per-channel pad value.')
      new_image = tf.stack(
          [
              tf.pad(
                  channels[i], [[0, max_dimension - new_size[0]],
                                [0, max_dimension - new_size[1]]],
                  constant_values=per_channel_pad_value[i])
              for i in range(len(channels))
          ],
          axis=2)
      new_image.set_shape([max_dimension, max_dimension, 3])

    result = [new_image]
    if masks is not None:
      new_masks = tf.expand_dims(masks, 3)
      new_masks = tf.image.resize_images(
          new_masks,
          new_size[:-1],
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
          align_corners=align_corners)
      if pad_to_max_dimension:
        new_masks = tf.image.pad_to_bounding_box(
            new_masks, 0, 0, max_dimension, max_dimension)
      new_masks = tf.squeeze(new_masks, 3)
      result.append(new_masks)

    result.append(new_size)
    return result

def np_resize_to_range(img,min_dimension,max_dimension=-1):
    new_shape = list(img.shape[:2])
    if img.shape[0]<img.shape[1]:
        new_shape[0] = min_dimension
        if max_dimension>0:
            new_shape[1] = min(int(new_shape[0]*img.shape[1]/img.shape[0]),max_dimension)
        else:
            new_shape[1] = int(new_shape[0]*img.shape[1]/img.shape[0])
    else:
        new_shape[1] = min_dimension
        if max_dimension>0:
            new_shape[0] = min(int(new_shape[1]*img.shape[0]/img.shape[1]),max_dimension)
        else:
            new_shape[0] = int(new_shape[1]*img.shape[0]/img.shape[1])

    return resize_img(img,new_shape)

def random_perm_channel(img,seed=None):
    with tf.name_scope("random_perm_channel"):
        channel_nr = img.get_shape().as_list()[-1]
        index = list(range(channel_nr))
        indexs = list(itertools.permutations(index,channel_nr))
        assert np.alltrue(np.array(indexs[0]) == np.array(index)), "error"
        indexs = indexs[1:]
        x = tf.random_shuffle(indexs,seed=seed)
        x = x[0,:]
        if len(img.get_shape()) == 3:
            img = tf.transpose(img,perm=[2,0,1])
            img = tf.gather(img,x)
            img = tf.transpose(img,perm=[1,2,0])
        elif len(img.get_shape()) == 4:
            img = tf.transpose(img,perm=[3,0,1,2])
            img = tf.gather(img,x)
            img = tf.transpose(img,perm=[1,2,3,0])
        else:
            print("Error channel.")
        return img

def psnr(labels,predictions,max_v = 2,scope=None):
    with tf.name_scope(scope,default_name="psnr"):
        loss1 = tf.losses.mean_squared_error(labels=labels,predictions=predictions,loss_collection=None)
        return tf.minimum(100.0,tf.cond(tf.greater(loss1,1e-6),lambda:10*tf.log(max_v**2/loss1)/np.log(10),lambda:100.0))

def nppsnr(labels,predictions,max_v = 2):
    loss1 = np.mean(np.square(np.array(labels-predictions).astype(np.float32)))
    if loss1<1e-6:
        return 100.0
    return 10*np.log(max_v**2/loss1)/np.log(10)


def rot90(image,clockwise=True):
    if clockwise:
        k = 1
    else:
        k = 3
    if isinstance(image,list):
        image = [tf.cond(tf.reduce_min(tf.shape(img))>0,lambda:tf.image.rot90(img,k),lambda:img) for img in image]
    else:
        image = tf.image.rot90(image,k)
    return image


def random_rot90(image,clockwise=None):
    if clockwise is None:
        return wmlt.probability_case([(0.34,lambda: image),(0.33,lambda: rot90(image,True)),(0.33,lambda:rot90(image,False))])
    else:
        return wmlt.probability_case([(0.5,lambda: image),(0.5,lambda: rot90(image,clockwise))])

def random_saturation(image,gray_image=None,minval=0.0,maxval=1.0,scope=None):
    with tf.name_scope(scope, 'random_saturation', [image]):
        if gray_image is None:
            gray_image = rgb_to_grayscale(image,keep_channels=True)
        ratio = tf.random_uniform(shape=(),
                                  minval=minval,
                                  maxval=maxval,
                                  dtype=tf.float32,
                                  seed=int(time.time()))
        return gray_image*ratio + image*(1.0-ratio)

class ImagePatch(object):
    def __init__(self,patch_size):
        self.patch_size = patch_size
        self.patchs = None
        self.batch_size = None
        self.height = None
        self.width = None
        self.channel = None
    '''
    将图像[batch_size,height,width,channel]变换为[X,patch_size,patch_size,channel]
    '''
    def to_patch(self,images,scope=None):
        with tf.name_scope(scope,"to_patch"):
            patch_size = self.patch_size
            batch_size, height, width, channel = wmlt.combined_static_and_dynamic_shape(images)
            self.batch_size, self.height, self.width, self.channel = batch_size, height, width, channel
            net = tf.reshape(images, [batch_size, height // patch_size, patch_size, width // patch_size, patch_size,
                                   channel])
            net = tf.transpose(net, [0, 1, 3, 2, 4, 5])
            self.patchs = tf.reshape(net, [-1, patch_size, patch_size, channel])
            return self.patchs

    def from_patch(self,scope=None):
        assert self.patchs is not None,"Must call to_path first."
        with tf.name_scope(scope,"from_patch"):
            batch_size, height, width, channel = self.batch_size, self.height, self.width, self.channel
            patch_size = self.patch_size
            net = tf.reshape(self.patchs, [batch_size, height // patch_size, width // patch_size, patch_size, patch_size,
                                   channel])
            net = tf.transpose(net, [0, 1, 3, 2, 4, 5])
            net = tf.reshape(net, [batch_size, height, width, channel])
            return net

class NPImagePatch(object):
    def __init__(self,patch_size):
        self.patch_size = patch_size
        self.patchs = None
        self.batch_size = None
        self.height = None
        self.width = None
        self.channel = None
    '''
    将图像[batch_size,height,width,channel]变换为[X,patch_size,patch_size,channel]
    '''
    def to_patch(self,images):
        patch_size = self.patch_size
        batch_size, height, width, channel = images.shape
        self.batch_size, self.height, self.width, self.channel = batch_size, height, width, channel
        net = np.reshape(images, [batch_size, height // patch_size, patch_size, width // patch_size, patch_size,
                                  channel])
        net = np.transpose(net, [0, 1, 3, 2, 4, 5])
        self.patchs = np.reshape(net, [-1, patch_size, patch_size, channel])
        return self.patchs

    def from_patch(self):
        assert self.patchs is not None,"Must call to_path first."
        batch_size, height, width, channel = self.batch_size, self.height, self.width, self.channel
        patch_size = self.patch_size
        net = np.reshape(self.patchs, [batch_size, height // patch_size, width // patch_size, patch_size, patch_size,
                                       channel])
        net = np.transpose(net, [0, 1, 3, 2, 4, 5])
        net = np.reshape(net, [batch_size, height, width, channel])
        return net
