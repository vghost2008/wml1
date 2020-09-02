#coding=utf-8
import wmodule
from object_detection2.standard_names import *
import tensorflow as tf
import img_utils as wmli
import numpy as np
import object_detection2.bboxes as odb
import wml_tfutils as wmlt
from collections import OrderedDict
import wsummary

class MetaArch(wmodule.WModule):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
    @staticmethod
    def m0v1(image):
        """Normalize the image to zero mean and unit variance."""
        # The image normalization is identical to Cloud TPU ResNet.
        image = image/255.0
        offset = tf.constant([0.485, 0.456, 0.406])
        offset = tf.expand_dims(offset, axis=0)
        offset = tf.expand_dims(offset, axis=0)
        image -= offset
    
        scale = tf.constant([0.229, 0.224, 0.225])
        scale = tf.expand_dims(scale, axis=0)
        scale = tf.expand_dims(scale, axis=0)
        image /= scale
        return image
    
    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        with tf.name_scope(f"preprocess_image_{self.cfg.MODEL.PREPROCESS}"):
            b_img = batched_inputs[IMAGE]
            if self.cfg.MODEL.PREPROCESS == "ton1p1":
                b_img = (b_img-127.5)/127.5
            elif self.cfg.MODEL.PREPROCESS == "m0v1":
                b_img = MetaArch.m0v1(b_img)
            elif self.cfg.MODEL.PREPROCESS == "subimagenetmean":
                channel_means = [123.68, 116.779, 103.939]
                b_img = b_img-[[[channel_means]]]
            elif self.cfg.MODEL.PREPROCESS == "standardization":
                b_img = tf.image.per_image_standardization(b_img)
            elif self.cfg.MODEL.PREPROCESS == "standardization":
                pass
            else:
                raise ValueError(f"Error preprocess type {self.cfg.MODEL.PREPROCESS}")
            wsummary.histogram_or_scalar(b_img,"preprocessed_image")
            batched_inputs[IMAGE] = b_img
            return batched_inputs

    def _postprocess(self,instances, batched_inputs):
        if self.cfg.MODEL.MIN_BOXES_AREA_TEST>1e-5:
            with tf.name_scope("postprocess"):
                instances[RD_BOXES] = tf.clip_by_value(instances[RD_BOXES],0.0,1.0)
                box_are = odb.box_area(instances[RD_BOXES])
                mask = tf.greater(box_are,self.cfg.MODEL.MIN_BOXES_AREA_TEST)
                size = tf.shape(instances[RD_LABELS])[1]
                mask0 = tf.sequence_mask(instances[RD_LENGTH],maxlen=size)
                mask = tf.logical_and(mask,mask0)
                res = OrderedDict()
                del instances[RD_LENGTH]
                for k,v in instances.items():
                    n_v = wmlt.batch_boolean_mask(v,mask,size=size)
                    res[k] = n_v
                length = tf.reduce_sum(tf.cast(mask,tf.int32),axis=1)
                res[RD_LENGTH] = length
            return res
        else:
            return instances

    def doeval(self,evaler,datas):
        assert datas[GT_BOXES].shape[0]==1,"Error batch size"
        kwargs = {}
        image = datas[IMAGE][0]
        gt_boxes = datas[GT_BOXES][0]
        gt_labels = datas[GT_LABELS][0]
        len = datas[RD_LENGTH][0]
        boxes = datas[RD_BOXES][0][:len]
        probability = datas[PD_PROBABILITY][0][:len]
        labels = datas[RD_LABELS][0][:len]

        kwargs['gtboxes'] = gt_boxes
        kwargs['gtlabels'] = gt_labels
        kwargs['boxes'] = boxes
        kwargs['labels'] = labels
        kwargs['probability'] = probability
        '''if min(image.shape[0],image.shape[1])>self.cfg.INPUT.MIN_SIZE_TEST:
            h = image.shape[0]
            w = image.shape[1]
            scale = self.cfg.INPUT.MIN_SIZE_TEST/min(w,h)
            kwargs['img_size'] = [h*scale,w*scale]
        else:
            kwargs['img_size'] = image.shape[0:2]'''
        kwargs['img_size'] = image.shape[0:2]

        if RD_MASKS in datas and GT_MASKS in datas:
            gt_masks = datas[GT_MASKS][0]
            masks = datas[RD_MASKS][0][:len]
            N,H,W = masks.shape
            croped_gt_masks = wmli.one_to_one_crop_and_resize_imgs(gt_masks,gt_boxes,crop_size=[H,W])
            croped_gt_masks = (croped_gt_masks+0.1).astype(np.uint8)
            kwargs['gtmasks'] = croped_gt_masks
            '''
            for i in range(gt_masks.shape[0]):
                wmli.imsave(wmlu.home_dir(f"IMG_{i}_GT.jpg"),gt_masks[i]*255)
                wmli.imsave(wmlu.home_dir(f"IMG_{i}_CROPED_GT.jpg"),croped_gt_masks[i]*255)
            for i in range(masks.shape[0]):
                wmli.imsave(wmlu.home_dir(f"IMG_{i}_PRED.jpg"),masks[i]*255)
            '''
            if (masks.dtype != np.uint8) and (masks.dtype != np.bool):
                masks = (masks>0.5).astype(np.uint8)
            if masks.dtype == np.bool:
                masks = masks.astype(np.uint8)
            kwargs['masks'] = masks

        evaler(**kwargs)
