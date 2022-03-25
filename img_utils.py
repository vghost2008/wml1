#coding=utf-8
#import pydicom as dcm
import PIL.Image
from PIL import Image,ImageOps
import io
import scipy.misc
import matplotlib.image as mpimg
import numpy as np
import shutil
import os
import cv2
import tensorflow as tf
import copy
import random
import itertools
import time
import basic_tftools as btf
import glob
from collections import OrderedDict
from object_detection2.basic_datadef import DEFAULT_COLOR_MAP as _DEFAULT_COLOR_MAP
import object_detection2.visualization as odv

try:
    from turbojpeg import TJCS_RGB, TJPF_BGR, TJPF_GRAY, TurboJPEG
except ImportError:
    TJCS_RGB = TJPF_GRAY = TJPF_BGR = TurboJPEG = None
g_jpeg = None

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

def nprgb_to_gray(img,keep_channels=False):
    img_gray = img * np.array([0.299, 0.587, 0.114], dtype=np.float32)
    img_gray = np.sum(img_gray,axis=-1)
    if keep_channels:
        img_gray = np.stack([img_gray,img_gray,img_gray],axis=-1)
    return img_gray

def npbatch_rgb_to_gray(img,keep_channels=False):
    if not isinstance(img,np.ndarray):
        img = np.array(img)
    img_gray = img * np.array([0.299, 0.587, 0.114], dtype=np.float32)
    img_gray = np.sum(img_gray,axis=-1)
    if keep_channels:
        img_gray = np.stack([img_gray,img_gray,img_gray],axis=-1)
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
'''
size:(w,h)
'''
def resize_img(img,size,keep_aspect_ratio=False,interpolation=cv2.INTER_LINEAR,align=None):

    img_shape = img.shape
    if size[0] == img.shape[1] and size[1]==img.shape[0]:
        return img

    if np.any(np.array(img_shape)==0):
        img_shape = list(img_shape)
        img_shape[0] = size[1]
        img_shape[1] = size[0]
        return np.zeros(img_shape,dtype=img.dtype)
    if keep_aspect_ratio:
        if size[1]*img_shape[1] != size[0]*img_shape[0]:
            if size[1]*img_shape[1]>size[0]*img_shape[0]:
                ratio = size[0]/img_shape[1]
            else:
                ratio = size[1]/img_shape[0]
            size = list(copy.deepcopy(size))
            size[0] = int(img_shape[1]*ratio)
            size[1] = int(img_shape[0]*ratio)

            if align:
                size[0] = (size[0]+align-1)//align*align
                size[1] = (size[1] + align - 1) // align * align

    if not isinstance(size,tuple):
        size = tuple(size)
    if size[0]==img_shape[0] and size[1]==img_shape[1]:
        return img
    return cv2.resize(img,dsize=size,interpolation=interpolation)

def resize_height(img,h,interpolation=cv2.INTER_LINEAR):
    shape = img.shape
    new_h = h
    new_w = int(shape[1]*new_h/shape[0])
    return cv2.resize(img,dsize=(new_w,new_h),interpolation=interpolation)

def resize_width(img,w,interpolation=cv2.INTER_LINEAR):
    shape = img.shape
    new_w = w
    new_h = int(shape[0]*new_w/shape[1])
    return cv2.resize(img,dsize=(new_w,new_h),interpolation=interpolation)

def resize_short_size(img,size,interpolation=cv2.INTER_LINEAR):
    shape = img.shape
    if shape[0]<shape[1]:
        return resize_height(img,size,interpolation)
    else:
        return resize_width(img,size,interpolation)
'''
size:(w,h)
'''
def resize_and_pad(img,size,interpolation=cv2.INTER_LINEAR,pad_color=(0,0,0),center_pad=True,return_scale=False):
    old_shape = img.shape
    img = resize_img(img,size,keep_aspect_ratio=True,interpolation=interpolation)
    if return_scale:
        r = img.shape[0]/max(old_shape[0],1)
    if img.shape[0] == size[1] and img.shape[1] == size[0]:
        if return_scale:
            return img,r
        return img
    else:
        res = np.ones([size[1],size[0],3],dtype=img.dtype)
        pad_color = np.array(list(pad_color))
        pad_color = pad_color.reshape([1,1,3])
        res = res*pad_color
        if center_pad:
            offset_x = (size[0]-img.shape[1])//2
            offset_y = (size[1]-img.shape[0])//2
        else:
            offset_x = 0
            offset_y = 0

        w = img.shape[1]
        h = img.shape[0]
        res[offset_y:offset_y+h,offset_x:offset_x+w,:] = img
        if return_scale:
            return res,r
        else:
            return res

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
    ymin = int((shape[0])*box[0]+0.5)
    ymax = int((shape[0])*box[2]+1+0.5)
    xmin = int((shape[1])*box[1]+0.5)
    xmax = int((shape[1])*box[3]+1+0.5)
    if len(shape)==2:
        return img[ymin:ymax,xmin:xmax]
    else:
        return img[ymin:ymax,xmin:xmax,:]

'''
box:ymin,xmin,ymax,xmax, absolute corrdinate
'''
def crop_img_absolute(img,box):
    shape = img.shape
    box = [box[0]/shape[0],box[1]/shape[1],box[2]/shape[0],box[3]/shape[1]]
    return crop_img(img,box)
'''
box:[ymin,xmin,ymax,xmax], relative coordinate
crop_size:[heigh,width] absolute pixel size.
'''
def crop_and_resize(img,box,crop_size):
    img = crop_img(img,box)
    return resize_img(img,crop_size)

'''
img:[H,W]/[H,W,C]
box:[N,4] ymin,xmin,ymax,xmax, relative corrdinate
从同一个图上切图
'''
def crop_and_resize_imgs(img,boxes,crop_size):
    res_imgs = []
    for box in boxes:
        sub_img = crop_and_resize(img,box,crop_size)
        res_imgs.append(sub_img)

    return np.stack(res_imgs,axis=0)
'''
img:[N,H,W]/[N,H,W,C]
box:[N,4] ymin,xmin,ymax,xmax, relative corrdinate
box 与 img一对一的进行切图
return:
[N]+crop_size
'''
def one_to_one_crop_and_resize_imgs(imgs,boxes,crop_size):
    res_imgs = []
    for i,box in enumerate(boxes):
        sub_img = crop_and_resize(imgs[i],box,crop_size)
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

'''
images:list of image with shape [B,H,W,C]
'''
def concat_images(images,margin=10):
    with tf.name_scope("concat_images"):
        new_images = []
        mean = tf.reduce_mean(images[0])
        B,H,W,C = btf.combined_static_and_dynamic_shape(images[0])
        size = [H,W]
        if margin is not None:
            margin_img = tf.ones([B,H,margin,C],dtype=images[0].dtype)*mean
        for i,img in enumerate(images):
            if margin is not None and i>0:
                new_images.append(margin_img)
            new_images.append(tf.image.resize_bilinear(img,size=size))
        return tf.concat(new_images,axis=2)
'''
img:[H,W]/[H,W,C]
rect:[ymin,xmin,ymax,xmax] absolute coordinate
'''
def sub_image(img,rect,pad_value=127):
    if rect[0]<0 or rect[1]<0 or rect[2]>img.shape[0] or rect[3]>img.shape[1]:
        py0 = -rect[0] if rect[0]<0 else 0
        py1 = rect[2]-img.shape[0] if rect[2]>img.shape[0] else 0
        px0 = -rect[1] if rect[1] < 0 else 0
        px1 = rect[3] - img.shape[1] if rect[3] > img.shape[1] else 0
        img = np.pad(img,[[py0,py1],[px0,px1],[0,0]],constant_values=pad_value)
        rect[0] -= py0
        rect[1] -= px0

    return copy.deepcopy(img[rect[0]:rect[2],rect[1]:rect[3]])

'''
img:[H,W]/[H,W,C]
rect:[N,4] [ymin,xmin,ymax,xmax] absolute coordinate
'''
def sub_images(img,rects):
    res = []
    for rect in rects:
        res.append(sub_image(img,rect))

    return res
'''
img:[H,W]/[H,W,C]
rect:[xmin,ymin,xmax,ymax] absolute coordinate
'''
def sub_imagev2(img,rect,pad_value=127):
    return sub_image(img,[rect[1],rect[0],rect[3],rect[2]],pad_value=pad_value)

'''
img: [H,W,C]
size: [w,h]
'''
def center_crop(img,size,pad_value=127):
    cx = img.shape[1]//2
    cy = img.shape[0]//2
    x0 = cx-size[0]//2
    y0 = cy-size[1]//2
    x1 = x0+size[0]
    y1 = y0+size[1]
    return sub_image(img,[y0,x0,y1,x1],pad_value=pad_value)
'''
img:[H,W,C]
size:(w,h)
'''
def pad_img(img,size,pad_value=127):
    if img.shape[0]<size[1]:
        py0 = (size[1]-img.shape[0])//2
        py1 = size[1]-img.shape[0]-py0
    else:
        py0 = 0
        py1 = 0
    if img.shape[1]<size[0]:
        px0 = (size[0] - img.shape[1]) // 2
        px1 = size[0] - img.shape[1] - px0
    else:
        px0 = 0
        px1 = 0
    img = np.pad(img, [[py0, py1], [px0, px1], [0, 0]], constant_values=pad_value)
    return img


'''
img:[H,W]/[H,W,C]
rect:[N,4] [xmin,ymin,xmax,ymax] absolute coordinate
'''
def sub_imagesv2(img,rects):
    res = []
    for rect in rects:
        res.append(sub_imagev2(img,rect))

    return res

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
    dir_path = os.path.dirname(filename)
    if dir_path != "" and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if len(img.shape)==3 and img.shape[2]==3:
        img = copy.deepcopy(img)
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR,img)
    cv2.imwrite(filename, img)

def read_and_write_img(src_path,dst_path):
    img = cv2.imread(src_path)
    cv2.imwrite(dst_path,img)

def imwrite_mask(filename,mask,color_map=_DEFAULT_COLOR_MAP):
    if os.path.splitext(filename)[1].lower() != ".png":
        print("WARNING: mask file need to be png format.")
    if not isinstance(mask,np.ndarray):
        mask = np.ndarray(mask)
    if len(mask.shape)==3:
        if mask.shape[-1] != 1:
            raise RuntimeError(f"ERROR mask shape {mask.shape}")
        mask = np.squeeze(mask,axis=-1)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(color_map)
    new_mask.save(filename)

def imwrite_mask_on_img(filename,img,mask,color_map=_DEFAULT_COLOR_MAP,ignored_label=255):
    r_img = odv.draw_semantic_on_image(img,mask, color_map, ignored_label=ignored_label)
    imwrite(filename,r_img)

def imread_mask(filename):
    mask = Image.open(filename)
    return np.array(mask)

def videowrite(filename,imgs,fps=30,fmt="RGB"):
    if len(imgs)==0:
        return
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    write_size = imgs[0].shape[:2][::-1]
    video_writer = cv2.VideoWriter(filename, fourcc, fps,write_size)
    if fmt == "BGR":
        for img in imgs:
            video_writer.write(img)
    elif fmt=="RGB":
        for img in imgs:
            video_writer.write(img[...,::-1])
    else:
        print(f"ERROR fmt {fmt}.")
    video_writer.release()

class VideoWriter:
    def __init__(self,filename,fps=30,fmt='RGB'):
        self.video_writer = None
        self.fmt = fmt
        self.fps = fps
        self.filename = filename

    def __del__(self):
        self.release()

    def init_writer(self,img):
        if self.video_writer is not None:
            return
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        write_size = img.shape[:2][::-1]
        self.video_writer = cv2.VideoWriter(self.filename, fourcc, self.fps, write_size)

    def write(self,img):
        if self.video_writer is None:
            self.init_writer(img)
        fmt = self.fmt
        if fmt == "BGR":
            self.video_writer.write(img)
        elif fmt=="RGB":
            self.video_writer.write(img[...,::-1])
        else:
            print(f"ERROR fmt {fmt}.")

    def release(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

class VideoReader:
    def __init__(self,path,file_pattern="img_{:05d}.jpg",suffix=".jpg",preread_nr=0) -> None:
        if os.path.isdir(path):
            self.dir_path = path
            self.reader = None
            self.all_files = glob.glob(os.path.join(path,"*"+suffix))
            self.frames_nr = len(self.all_files)
            self.fps = 1
        else:
            self.reader = cv2.VideoCapture(path)
            self.dir_path = None
            self.frames_nr = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.reader.get(cv2.CAP_PROP_FPS)
            self.preread_nr = preread_nr
            if self.preread_nr>1:
                self.reader_buffer = OrderedDict()
            else:
                self.reader_buffer = None


        self.idx = 1
        self.file_pattern = file_pattern

    def __iter__(self):
        return self
    
    def __getitem__(self,idx):
        if self.dir_path is None:
            if self.preread_nr>1:
                if idx in self.reader_buffer:
                    return self.reader_buffer[idx]
                elif idx<self.idx-1:
                    raise NotImplemented()
                else:
                    for x in range(self.idx-1,idx+1):
                        if x in self.reader_buffer:
                            continue
                        ret,frame = self.reader.read()
                        if ret:
                            frame = frame[...,::-1]
                            self.reader_buffer[x] = frame
                    if idx in self.reader_buffer:
                        return self.reader_buffer[idx]

            raise NotImplemented()
        elif idx<self.frames_nr:
            if self.file_pattern is None:
                file_path = self.all_files[idx]
            else:
                file_path = os.path.join(self.dir_path,self.file_pattern.format(idx+1))
            img = cv2.imread(file_path)
            return img[...,::-1]
        else:
            raise RuntimeError()
    
    def __len__(self):
        if self.dir_path is not None:
            return self.frames_nr
        elif self.reader is not None:
            return self.frames_nr
        else:
            raise RuntimeError()

    def __next__(self):
        if self.reader is not None:
            if self.preread_nr>1:
                if self.idx-1 in self.reader_buffer:
                    frame = self.reader_buffer[self.idx-1]
                    ret = True
                else:
                    ret,frame = self.reader.read()
                    if not ret:
                        raise StopIteration()
                    frame = frame[...,::-1]
                    self.reader_buffer[self.idx-1] = frame
                    while len(self.reader_buffer)>self.preread_nr:
                        self.reader_buffer.popitem(last=False)
            else:
                ret,frame = self.reader.read()
                if ret:
                    frame = frame[...,::-1]
            
            self.idx += 1
            if not ret:
                raise StopIteration()
            else:
                return frame
        else:
            if self.idx>self.frames_nr:
                raise StopIteration()
            if self.file_pattern is not None:
                file_path = os.path.join(self.dir_path,self.file_pattern.format(self.idx))
            else:
                file_path = self.all_files[self.idx-1]
            img = cv2.imread(file_path)
            self.idx += 1
            return img[...,::-1]


def rotate_img(img,angle,scale=1.0):
    center = (img.shape[1]//2,img.shape[0]//2)
    M = cv2.getRotationMatrix2D(center,angle,scale)
    img = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
    return img

def rotate_img_file(filepath,angle,scale=1.0):
    img = cv2.imread(filepath)
    center = (img.shape[1]//2,img.shape[0]//2)
    M = cv2.getRotationMatrix2D(center,angle,scale)
    img = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
    cv2.imwrite(filepath,img)

def imshow(winname,img):
    img = copy.deepcopy(img)
    cv2.cvtColor(img,cv2.COLOR_RGB2BGR,img)
    cv2.imshow(winname,img)

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
        return btf.probability_case([(0.34,lambda: image),(0.33,lambda: rot90(image,True)),(0.33,lambda:rot90(image,False))])
    else:
        return btf.probability_case([(0.5,lambda: image),(0.5,lambda: rot90(image,clockwise))])

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
            batch_size, height, width, channel = btf.combined_static_and_dynamic_shape(images)
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

    def from_patch(self,patchs=None):
        assert self.patchs is not None,"Must call to_path first."
        if patchs is not None:
            self.patchs = patchs
        batch_size, height, width, channel = self.batch_size, self.height, self.width, self.channel
        patch_size = self.patch_size
        net = np.reshape(self.patchs, [batch_size, height // patch_size, width // patch_size, patch_size, patch_size,
                                       channel])
        net = np.transpose(net, [0, 1, 3, 2, 4, 5])
        net = np.reshape(net, [batch_size, height, width, channel])
        return net

'''
bboxes:[N,4],[ymin,xmin,ymax,xmax], absolute coordinate
'''
def remove_boxes_of_img(img,bboxes,default_value=[127,127,127]):
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes)
    if bboxes.shape[0] == 0:
        return img
    ymin,xmin,ymax,xmax = np.transpose(bboxes)
    ymin = np.maximum(ymin,0)
    xmin = np.maximum(xmin,0)
    ymax = np.minimum(ymax,img.shape[0])
    xmax = np.minimum(xmax,img.shape[1])
    bboxes = np.stack([ymin,xmin,ymax,xmax],axis=1)

    for box in bboxes:
        img[box[0]:box[2], box[1]:box[3]] = default_value
    return img

def img_info(img):
    if len(img.shape) == 3 and img.shape[-1]>1:
        img = nprgb_to_gray(img)
    return np.std(img)

def tf_img_info(img):
    img = tf.image.rgb_to_grayscale(img)
    if img.dtype != tf.float32:
        img = tf.cast(img,tf.float32)
    mean,variance = tf.nn.moments(img,list(range(len(img.get_shape()))))
    return variance

'''
img: np.ndarray, [H,W,3], RGB order
return:
bytes of jpeg string
'''
def encode_img(img,quality=95):
    pil_image = PIL.Image.fromarray(img)
    output_io = io.BytesIO()
    pil_image.save(output_io, format='JPEG',quality=quality)
    return output_io.getvalue()

def _jpegflag(flag='color', channel_order='bgr'):
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'color':
        if channel_order == 'bgr':
            return TJPF_BGR
        elif channel_order == 'rgb':
            return TJCS_RGB
    elif flag == 'grayscale':
        return TJPF_GRAY
    else:
        raise ValueError('flag must be "color" or "grayscale"')

def decode_img(buffer):
    if TurboJPEG is not None:
        global g_jpeg
        if g_jpeg is None:
            g_jpeg = TurboJPEG()
        img = g_jpeg.decode(buffer,TJCS_RGB)
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        return img
    buff = io.BytesIO(buffer)
    img = PIL.Image.open(buff)

    img = pillow2array(img, 'color')

    return img

def pillow2array(img,flag='color'):
    # Handle exif orientation tag
    if flag in ['color', 'grayscale']:
        img = ImageOps.exif_transpose(img)
    # If the image mode is not 'RGB', convert it to 'RGB' first.
    if img.mode != 'RGB':
        if img.mode != 'LA':
            # Most formats except 'LA' can be directly converted to RGB
            img = img.convert('RGB')
        else:
            # When the mode is 'LA', the default conversion will fill in
            #  the canvas with black, which sometimes shadows black objects
            #  in the foreground.
            #
            # Therefore, a random color (124, 117, 104) is used for canvas
            img_rgba = img.convert('RGBA')
            img = Image.new('RGB', img_rgba.size, (124, 117, 104))
            img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
    if flag in ['color', 'color_ignore_orientation']:
        array = np.array(img)
    elif flag in ['grayscale', 'grayscale_ignore_orientation']:
        img = img.convert('L')
        array = np.array(img)
    else:
        raise ValueError(
            'flag must be "color", "grayscale", "unchanged", '
            f'"color_ignore_orientation" or "grayscale_ignore_orientation"'
            f' but got {flag}')
    return array

