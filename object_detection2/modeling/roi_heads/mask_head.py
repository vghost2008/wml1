#coding=utf-8
from thirdparty.registry import Registry
import wmodule
import tensorflow as tf
import wml_tfutils as wmlt
from object_detection2.datadef import *
import object_detection2.config.config as config
import image_visualization as ivis
import wsummary
import img_utils as wmli

slim = tf.contrib.slim

ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@wmlt.add_name_scope
def mask_rcnn_loss(inputs,pred_mask_logits, proposals:EncodedData,fg_selection_mask):
    '''

    :param inputs:
    :param pred_mask_logits: [X,H,W,C] C==1 if cls_anostic_mask else num_classes
    :param proposals:
    :param fg_selection_mask: [X]
    :return:
    '''
    cls_agnostic_mask = pred_mask_logits.get_shape().as_list()[-1] == 1
    total_num_masks,mask_H,mask_W,C  = wmlt.combined_static_and_dynamic_shape(pred_mask_logits)
    assert mask_H==mask_W, "Mask prediction must be square!"

    gt_masks = inputs[GT_MASKS] #[batch_size,X,H,W]
    batch_size,X,H,W = wmlt.combined_static_and_dynamic_shape(gt_masks)
    gt_masks = wmlt.batch_gather(gt_masks,proposals.indices)
    gt_masks = tf.reshape(gt_masks,[-1,H,W])
    gt_masks = tf.boolean_mask(gt_masks,fg_selection_mask)
    boxes = proposals.boxes
    batch_size,box_nr,box_dim = wmlt.combined_static_and_dynamic_shape(boxes)
    boxes = tf.reshape(boxes,[batch_size*box_nr,box_dim])
    boxes = tf.boolean_mask(boxes,fg_selection_mask)
    gt_masks = tf.expand_dims(gt_masks,axis=-1)
    croped_masks_gt_masks = wmlt.tf_crop_and_resize(gt_masks,boxes,[mask_H,mask_W])

    if not cls_agnostic_mask:
        gt_classes = proposals.gt_object_logits
        gt_classes = tf.reshape(gt_classes,[-1])
        gt_classes = tf.boolean_mask(gt_classes,fg_selection_mask)
        pred_mask_logits = tf.transpose(pred_mask_logits,[0,3,1,2])
        pred_mask_logits = wmlt.batch_gather(pred_mask_logits,gt_classes-1) #预测中不包含背景
        pred_mask_logits = tf.expand_dims(pred_mask_logits,axis=-1)


    if config.global_cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
        with tf.name_scope("mask_loss_summary"):
            pmasks_2d = tf.reshape(fg_selection_mask,[batch_size,box_nr])
            boxes_3d = tf.expand_dims(boxes,axis=1)
            wsummary.positive_box_on_images_summary(inputs[IMAGE],proposals.boxes,
                                                    pmasks=pmasks_2d)
            image = wmlt.select_image_by_mask(inputs[IMAGE],pmasks_2d)
            t_gt_masks = tf.expand_dims(tf.squeeze(gt_masks,axis=-1),axis=1)
            wsummary.detection_image_summary(images=image,boxes=boxes_3d,instance_masks=t_gt_masks,
                                             name="mask_and_boxes_in_mask_loss")
            log_mask = gt_masks
            log_mask = ivis.draw_detection_image_summary(log_mask,boxes=tf.expand_dims(boxes,axis=1))
            log_mask = wmli.concat_images([log_mask, croped_masks_gt_masks])
            wmlt.image_summaries(log_mask,"mask",max_outputs=32)

            log_mask = wmli.concat_images([gt_masks, tf.cast(pred_mask_logits>0.5,tf.float32)])
            wmlt.image_summaries(log_mask,"gt_vs_pred",max_outputs=32)

    mask_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=croped_masks_gt_masks,logits=pred_mask_logits)
    mask_loss = tf.reduce_mean(mask_loss)

    return mask_loss
    pass

@wmlt.add_name_scope
def mask_rcnn_inference(pred_mask_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.get_shape().as_list()[-1] == 1
    labels = pred_instances[RD_LABELS]
    batch_size,box_nr = wmlt.combined_static_and_dynamic_shape(labels)
    if not cls_agnostic_mask:
        # Select masks corresponding to the predicted classes
        pred_mask_logits = tf.transpose(pred_mask_logits,[0,3,1,2])
        labels = tf.reshape(labels,[-1])-1 #去掉背景
        pred_mask_logits = wmlt.batch_gather(pred_mask_logits,labels)
    total_box_nr,H,W = wmlt.combined_static_and_dynamic_shape(pred_mask_logits)
    pred_mask_logits = tf.reshape(pred_mask_logits,[batch_size,box_nr,H,W])

    pred_mask_logits = tf.nn.sigmoid(pred_mask_logits)

    pred_instances[RD_MASKS] = pred_mask_logits


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(wmodule.WChildModule):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg,**kwargs):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(MaskRCNNConvUpsampleHead, self).__init__(cfg,**kwargs)
        self.norm_params = {
            'decay': 0.997,
            'epsilon': 1e-4,
            'scale': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'fused': None,  # Use fused batch norm if possible.
            'is_training':self.is_training
        }
        #Detectron2默认没有使用normalizer
        self.normalizer_fn = slim.batch_norm

    def forward(self, x):
        cfg = self.cfg
        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on
        num_mask_classes = 1 if cls_agnostic_mask else num_classes

        with tf.variable_scope("MaskHead"):
            for k in range(num_conv):
                x = slim.conv2d(x,conv_dims,[3,3],padding="SAME",
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=self.normalizer_fn,
                                    normalizer_params=self.norm_params,
                                    scope=f"Conf{k}")
            x = slim.conv2d_transpose(x,conv_dims,kernel_size=2,
                                    stride=2,activation_fn=tf.nn.relu,
                                    scope="Upsample")
            x = slim.conv2d(x,num_mask_classes,kernel_size=1,activation_fn=None,normalizer_fn=None,
                            scope="predictor")
        return x


def build_mask_head(cfg,*args,**kwargs):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg,*args,**kwargs)