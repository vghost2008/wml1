#coding=utf-8
import tensorflow as tf
import wmodule
from basic_tftools import channel
from .build import META_ARCH_REGISTRY
from object_detection2.modeling.build import build_outputs
from object_detection2.modeling.backbone.build import build_backbone
from object_detection2.modeling.anchor_generator import build_anchor_generator
from object_detection2.modeling.box_regression import CenterBox2BoxTransform
from object_detection2.modeling.matcher import Matcher
import math
from object_detection2.standard_names import *
from object_detection2.modeling.onestage_heads.retinanet_outputs import *
from .meta_arch import MetaArch
from object_detection2.datadef import *
import object_detection2.od_toolkit as odtk
import wnnlayer as wnnl
from functools import partial

slim = tf.contrib.slim

__all__ = ["CenterNet"]

def left_pool(x,kernel=13):
    x = tf.pad(x,paddings=[[0,0],[0,0],[0,kernel-1],[0,0]])
    x = slim.max_pool2d(x,[1,kernel],stride=1,padding="VALID")
    return x

def right_pool(x,kernel=13):
    x = tf.pad(x,paddings=[[0,0],[0,0],[kernel-1,0],[0,0]])
    x = slim.max_pool2d(x,[1,kernel],stride=1,padding="VALID")
    return x

def top_pool(x,kernel=13):
    x = tf.pad(x,paddings=[[0,0],[kernel-1,0],[0,0],[0,0]])
    x = slim.max_pool2d(x,[kernel,1],stride=1,padding="VALID")
    return x

def bottom_pool(x,kernel=13):
    x = tf.pad(x,paddings=[[0,0],[0,kernel-1],[0,0],[0,0]])
    x = slim.max_pool2d(x,[kernel,1],stride=1,padding="VALID")
    return x

@META_ARCH_REGISTRY.register()
class CenterNet(MetaArch):
    """
    Implement CenterNet (https://arxiv.org/abs/1708.02002).
    """

    def __init__(self, cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)

        # fmt: off
        self.num_classes              = cfg.MODEL.CENTERNET.NUM_CLASSES
        self.in_features              = cfg.MODEL.CENTERNET.IN_FEATURES
        self.k = cfg.MODEL.CENTERNET.K
        # fmt: on

        self.backbone = build_backbone(cfg,parent=self,*args,**kwargs)

        self.head = CenterNetHead(cfg=cfg.MODEL.CENTERNET,
                                  parent=self,
                                  *args,**kwargs)

        # Matching and loss
        self.box2box_transform = CenterBox2BoxTransform(num_classes=self.num_classes,k=self.k)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (H, W, C) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        batched_inputs = self.preprocess_image(batched_inputs)

        features = self.backbone(batched_inputs)
        if len(self.in_features) == 0:
            print(f"Error no input features for retinanet, use all features {features.keys()}")
            features = list(features.values())
        else:
            features = [features[f] for f in self.in_features]
        head_outputs = self.head(features)
        gt_boxes = batched_inputs[GT_BOXES]
        gt_length = batched_inputs[GT_LENGTH]
        gt_labels = batched_inputs[GT_LABELS]

        outputs = build_outputs(name=self.cfg.MODEL.CENTERNET.OUTPUTS,
            cfg=self.cfg.MODEL.CENTERNET,
            parent=self,
            box2box_transform=self.box2box_transform,
            head_outputs = head_outputs,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_length=gt_length,
            max_detections_per_image=self.cfg.TEST.DETECTIONS_PER_IMAGE,
        )

        if self.cfg.GLOBAL.SUMMARY_LEVEL <= SummaryLevel.DEBUG:
            for i, t_outputs in enumerate(head_outputs):
                wsummary.feature_map_summary(t_outputs['heatmaps_tl'], f'heatmaps_tl{i}')
                wsummary.feature_map_summary(t_outputs['heatmaps_br'], f'heatmaps_br{i}')
                wsummary.feature_map_summary(t_outputs['heatmaps_ct'], f'heatmaps_ct{i}')

        if self.is_training:
            if self.cfg.GLOBAL.SUMMARY_LEVEL<=SummaryLevel.DEBUG:
                results = outputs.inference(inputs=batched_inputs,head_outputs=head_outputs)
            else:
                results = {}

            return results,outputs.losses()
        else:
            results = outputs.inference(inputs=batched_inputs,
                                        head_outputs=head_outputs)
            return results,{}


class CenterNetHead(wmodule.WChildModule):
    """
    The head used in CenterNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg,parent,*args,**kwargs):
        '''

        :param cfg:  only the child part
        :param parent:
        :param args:
        :param kwargs:
        '''
        super().__init__(cfg,*args,parent=parent,**kwargs)
        # Detectron2默认没有使用normalizer, 但在测试数据集上发现不使用normalizer网络不收敛
        self.normalizer_fn,self.norm_params = odtk.get_norm(self.cfg.NORM,is_training=self.is_training)
        self.activation_fn = odtk.get_activation_fn(self.cfg.ACTIVATION_FN)
        self.norm_scope_name = odtk.get_norm_scope_name(self.cfg.NORM)

        '''self.left_pool = wop.left_pool
        self.right_pool = wop.right_pool
        self.bottom_pool = wop.bottom_pool
        self.top_pool = wop.top_pool'''
        '''self.left_pool = partial(wnnl.cnn_self_hattenation,scope="left_pool")
        self.right_pool = partial(wnnl.cnn_self_hattenation,scope="right_pool")
        self.bottom_pool = partial(wnnl.cnn_self_vattenation,scope="bottom_pool")
        self.top_pool = partial(wnnl.cnn_self_vattenation,scope="top_pool")'''
        self.left_pool = left_pool
        self.right_pool = right_pool
        self.bottom_pool = bottom_pool
        self.top_pool = top_pool

    def forward(self, features,reuse=None):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, Hi, Wi,AxK).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Hi, Wi, Ax4).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        cfg = self.cfg
        num_classes      = cfg.NUM_CLASSES
        all_outs = []
        for ind,feature in enumerate(features):
            with slim.arg_scope([slim.conv2d], activation_fn=self.activation_fn,
                                normalizer_fn=self.normalizer_fn,
                                normalizer_params=self.norm_params):
                with tf.variable_scope("shared_head",reuse=reuse) as scope:
                    if ind >0:
                        scope.reuse_variables()
                    tl_conv = self.tl_pool(feature)
                    br_conv = self.br_pool(feature)
                    ct_conv = self.center_pool(feature)
                    tl_heat = self.heat(tl_conv,out_dim=num_classes,scope='heat_tl')
                    br_heat = self.heat(br_conv,out_dim=num_classes,scope='heat_br')
                    ct_heat = self.heat(ct_conv,out_dim=num_classes,scope='heat_ct')
                    tl_tag = self.tag(tl_conv,out_dim=1,scope="tl_tag")
                    br_tag = self.tag(br_conv,out_dim=1,scope="br_tag")

                    tl_regr = self.offset(tl_conv,out_dim=2,scope="tl_regr")
                    br_regr = self.offset(br_conv,out_dim=2,scope="br_regr")
                    ct_regr = self.offset(ct_conv,out_dim=2,scope="ct_regr")

                    outs = {}
                    outs["heatmaps_tl"] = tl_heat
                    outs["heatmaps_br"] = br_heat
                    outs["heatmaps_ct"] = ct_heat
                    outs["2d_offset_tl"] = tl_regr
                    outs['2d_offset_br'] = br_regr
                    outs['2d_offset_ct'] = ct_regr
                    outs["2d_tag_tl"] = tl_tag
                    outs['2d_tag_br'] = br_tag
                    shape = wmlt.combined_static_and_dynamic_shape(tl_regr)
                    outs["offset_tl"] = tf.reshape(tl_regr, [shape[0], -1, shape[3]])
                    outs['offset_br'] = tf.reshape(br_regr, [shape[0], -1, shape[3]])
                    outs['offset_ct'] = tf.reshape(ct_regr, [shape[0], -1, shape[3]])
                    shape = wmlt.combined_static_and_dynamic_shape(tl_tag)
                    outs["tag_tl"] = tf.reshape(tl_tag,[shape[0],-1,shape[3]])
                    outs['tag_br'] = tf.reshape(br_tag,[shape[0],-1,shape[3]])

            all_outs.append(outs)

        return all_outs

    def heat(self,inputs,out_dim,scope='heat'):
        #out_dim=80
        with tf.variable_scope(scope):
            input_dim = inputs.get_shape().as_list()[-1]
            x=slim.conv2d(inputs,input_dim,[3,3],)
            x=slim.conv2d(x,out_dim,1,activation_fn=None,normalizer_fn=None)
            return x

    #tag is embeading
    def tag(self,inputs,out_dim,scope='tag'):
        #out_dim=1
        with tf.variable_scope(scope):
            input_dim = inputs.get_shape().as_list()[-1]
            x=slim.conv2d(inputs,input_dim,[3,3],)
            x=slim.conv2d(x,out_dim,1,activation_fn=None,normalizer_fn=None)
            return x

    def offset(self,inputs,out_dim,scope='offset'):
        #out_dim=2
        with tf.variable_scope(scope):
            input_dim = inputs.get_shape().as_list()[-1]
            x=slim.conv2d(inputs,input_dim,[3,3],)
            x=slim.conv2d(x,out_dim,1,activation_fn=None,normalizer_fn=None)
            return x

    def pool(self,x,pool1_fn,pool2_fn,dim=128,scope=None):
        out_dim = channel(x)
        with tf.variable_scope(scope,default_name="pool"):
            with tf.variable_scope("pool1"):
                look_conv1 = slim.conv2d(x,dim,3,
                                         rate=2)
                look_right = pool2_fn(look_conv1)

                p1_conv1 = slim.conv2d(x,dim,3,
                                       rate=2)
                p1_look_conv = slim.conv2d(p1_conv1+look_right,dim,3,
                                           biases_initializer=None)
                pool1 = pool1_fn(p1_look_conv)

            with tf.variable_scope("pool2"):
                look_conv2 = slim.conv2d(x, dim, 3,
                                        rate=2)

                look_down = pool1_fn(look_conv2)

                p2_conv1 = slim.conv2d(x, dim, 3,
                                       rate=2)
                p2_look_conv = slim.conv2d(p2_conv1 + look_down, dim, 3,
                                           biases_initializer=None)
                pool2 = pool2_fn(p2_look_conv)

                with tf.variable_scope("merge"):
                    p_conv1 = slim.conv2d(pool1+pool2,out_dim,3,biases_initializer=None,
                                          normalizer_fn=self.normalizer_fn,
                                          normalizer_params=self.norm_params,
                                          activation_fn=None)
                    conv1 = slim.conv2d(x,out_dim,1,
                                        biases_initializer=None,
                                        normalizer_fn=self.normalizer_fn,
                                        normalizer_params=self.norm_params,
                                        activation_fn=None)
                    if self.activation_fn is not None:
                        relu1 = self.activation_fn(p_conv1+conv1)
                    else:
                        relu1 = p_conv1 + conv1
                    conv2 = slim.conv2d(relu1,out_dim,3,
                                        normalizer_fn=None,
                                        activation_fn=None)
                    return conv2

    def pool_cross(self, x, pool1_fn, pool2_fn, pool3_fn, pool4_fn,dim=128, scope=None):
        out_dim = channel(x)
        with tf.variable_scope(scope, default_name="pool_cross"):
            x = slim.conv2d(x, dim, 3)
            x = slim.conv2d(x, dim, 3)
            x = slim.conv2d(x, out_dim, 3,
                            normalizer_fn=None,
                            activation_fn=None)
            return x
            '''with tf.variable_scope("pool1"):
                p1_conv1 = slim.conv2d(x, dim, 3,
                                       normalizer_fn=None,
                                       activation_fn=None)
                pool1 = pool1_fn(p1_conv1)
                pool1 = pool3_fn(pool1)

            with tf.variable_scope("pool2"):
                p2_conv1 = slim.conv2d(x, dim, 3,
                                       normalizer_fn=None,
                                       activation_fn=None)
                pool2 = pool2_fn(p2_conv1)
                pool2 = pool4_fn(pool2)

            with tf.variable_scope("merge"):
                p_conv1 = slim.conv2d(pool1+pool2, out_dim, 3,
                                    biases_initializer=None,
                                    normalizer_fn=self.normalizer_fn,
                                    normalizer_params=self.norm_params,
                                    activation_fn=None)

                conv1 = slim.conv2d(x, out_dim, 1,
                                    biases_initializer=None,
                                    normalizer_fn=self.normalizer_fn,
                                    normalizer_params=self.norm_params,
                                    activation_fn=None)
                if self.activation_fn is not None:
                    relu1 = self.activation_fn(p_conv1 + conv1)
                else:
                    relu1 = p_conv1+conv1
                conv2 = slim.conv2d(relu1, out_dim, 3,
                                    normalizer_fn=None,
                                    activation_fn=None)
                return conv2'''

    def tl_pool(self,x,scope="tl_pool"):
        return self.pool(x,self.top_pool,self.left_pool,scope=scope)

    def br_pool(self,x,scope="br_pool"):
        return self.pool(x,self.bottom_pool,self.right_pool,scope=scope)

    def center_pool(self,x,scope="center_pool"):
        return self.pool_cross(x,self.top_pool,self.left_pool,self.bottom_pool,self.right_pool,scope=scope)
