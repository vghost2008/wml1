import wmodule
import tensorflow as tf
import object_detection2.od_toolkit as odtk
from object_detection2.modeling.onestage_heads.build import ONESTAGE_HEAD
from object_detection2.config.config import global_cfg
import wnnlayer as wnnl
import functools

slim = tf.contrib.slim

@ONESTAGE_HEAD.register()
class FastFairMOTHead(wmodule.WChildModule):
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
        self.normalizer_fn,self.norm_params = odtk.get_norm(self.cfg.NORM,is_training=self.is_training)
        self.activation_fn = odtk.get_activation_fn(self.cfg.ACTIVATION_FN)
        self.norm_scope_name = odtk.get_norm_scope_name(self.cfg.NORM)
        self.head_conv_dim = self.cfg.HEAD_CONV_DIM
        self.conv_op = functools.partial(slim.separable_conv2d,
                                         depth_multiplier=1,
                                         normalizer_fn=self.normalizer_fn,
                                         normalizer_params=self.norm_params,
                                         activation_fn=self.activation_fn)

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
        num_classes      = global_cfg.MODEL.NUM_CLASSES
        all_outs = []
        for ind,feature in enumerate(features):
            with slim.arg_scope([slim.conv2d], activation_fn=self.activation_fn,
                                normalizer_fn=self.normalizer_fn,
                                normalizer_params=self.norm_params):
                with tf.variable_scope("shared_head",reuse=reuse) as scope:
                    if ind >0:
                        scope.reuse_variables()
                    net = feature
                    net0 = self.conv_op(net,self.head_conv_dim,3,scope="Detection")
                    net1 = self.conv_op(net,None,3,scope="Embedding")
                    ct_heat = self.head(net0,mid_dim=self.head_conv_dim,
                                        out_dim=num_classes,scope='heat_ct')
                    ct_regr = self.head(net0,mid_dim=self.head_conv_dim,
                                        out_dim=2,scope="ct_regr")
                    hw_regr = self.head(net0,mid_dim=self.head_conv_dim,
                                        out_dim=2,scope="hw_regr")
                    id_embedding = self.head(net1,mid_dim=self.head_conv_dim,
                                             out_dim=global_cfg.MODEL.MOT.FAIR_MOT_ID_DIM,
                                             scope="id_embedding")
                    id_embedding = tf.math.l2_normalize(id_embedding,axis=-1)
                    outs = {}
                    outs["heatmaps_ct"] = ct_heat
                    outs['offset'] = ct_regr
                    outs["hw"] = hw_regr
                    outs['id_embedding'] = id_embedding

            all_outs.append(outs)

        return all_outs

    def head(self,inputs,out_dim,mid_dim=256,scope='heat'):
        with tf.variable_scope(scope):
            input_dim = inputs.get_shape().as_list()[-1]
            x=self.conv_op(inputs,None,[3,3])
            x=slim.conv2d(x,out_dim,1,activation_fn=None,normalizer_fn=None,scope="Conv_1")
            return x
