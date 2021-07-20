import wmodule
import tensorflow as tf
import object_detection2.od_toolkit as odtk
from object_detection2.modeling.onestage_heads.build import ONESTAGE_HEAD
from object_detection2.config.config import global_cfg
import wnnlayer as wnnl
slim = tf.contrib.slim

@ONESTAGE_HEAD.register()
class FairMOTHead(wmodule.WChildModule):
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

    def forward(self, features,reuse=None):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
        """
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
                    ct_heat = self.head(net,mid_dim=self.head_conv_dim,
                                        out_dim=num_classes,scope='heat_ct')
                    ct_regr = self.head(net,mid_dim=self.head_conv_dim,
                                        out_dim=2,scope="ct_regr")
                    hw_regr = self.head(net,mid_dim=self.head_conv_dim,
                                        out_dim=2,scope="hw_regr")
                    id_embedding = self.head(net,mid_dim=self.head_conv_dim,
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
            x=slim.conv2d(inputs,mid_dim,[3,3])
            x=slim.conv2d(x,out_dim,1,activation_fn=None,normalizer_fn=None)
            return x
