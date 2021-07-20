import tensorflow as tf
import wmodule
import object_detection2.od_toolkit as odtk
import basic_tftools as btf
from .build import SEMANTIC_HEAD

slim = tf.contrib.slim


@SEMANTIC_HEAD.register()
class DeepLabHead(wmodule.WChildModule):
    def __init__(self,cfg,parent,*args,**kwargs):
        '''

        :param cfg: only child part
        :param parent:
        :param args:
        :param kwargs:
        '''
        super().__init__(cfg,parent=parent,*args,**kwargs)
        self.normalizer_fn, self.norm_params = odtk.get_norm(self.cfg.NORM, is_training=self.is_training)
        self.activation_fn = odtk.get_activation_fn(self.cfg.ACTIVATION_FN)


    def split_separable_conv2d(self,inputs,
                               filters,
                               kernel_size=3,
                               rate=1,
                               weight_decay=0.00004,
                               depthwise_weights_initializer_stddev=0.33,
                               pointwise_weights_initializer_stddev=0.06,
                               scope=None):
        """Splits a separable conv2d into depthwise and pointwise conv2d.

        This operation differs from `tf.layers.separable_conv2d` as this operation
        applies activation function between depthwise and pointwise conv2d.

        Args:
          inputs: Input tensor with shape [batch, height, width, channels].
          filters: Number of filters in the 1x1 pointwise convolution.
          kernel_size: A list of length 2: [kernel_height, kernel_width] of
            of the filters. Can be an int if both values are the same.
          rate: Atrous convolution rate for the depthwise convolution.
          weight_decay: The weight decay to use for regularizing the model.
          depthwise_weights_initializer_stddev: The standard deviation of the
            truncated normal weight initializer for depthwise convolution.
          pointwise_weights_initializer_stddev: The standard deviation of the
            truncated normal weight initializer for pointwise convolution.
          scope: Optional scope for the operation.

        Returns:
          Computed features after split separable conv2d.
        """
        outputs = slim.separable_conv2d(
            inputs,
            None,
            kernel_size=kernel_size,
            depth_multiplier=1,
            rate=rate,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=depthwise_weights_initializer_stddev),
            weights_regularizer=None,
            activation_fn=self.activation_fn,
            normalizer_fn=self.normalizer_fn,
            normalizer_params=self.norm_params,
            scope=scope + '_depthwise')
        return slim.conv2d(
            outputs,
            filters,
            1,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=pointwise_weights_initializer_stddev),
            weights_regularizer=slim.l2_regularizer(weight_decay),
            activation_fn=self.activation_fn,
            normalizer_fn=self.normalizer_fn,
            normalizer_params=self.norm_params,
            scope=scope + '_pointwise')

    def forward(self,features,reuse=None):
        '''

        :param features: list[Tensor] high to low resolution
        :return:
        '''

        with tf.variable_scope("DeepLabHead"):
            conv_dim = self.cfg.CONV_DIM #256
            weight_decay = self.cfg.WEIGHT_DECAY #4e-5
            fea = features[-1]
            branch_logits = []
            with tf.variable_scope("image_pooling"):
                size = btf.img_size(fea)
                fea = tf.reduce_mean(fea, axis=[1,2],
                                     keepdims=True)
                fea = slim.conv2d(fea,conv_dim,[1,1],
                                  activation_fn=self.activation_fn,
                                  normalizer_fn=self.normalizer_fn,
                                  normalizer_params=self.norm_params,
                                  biases_initializer=None if self.normalizer_fn is not None else tf.zeros_initializer(),
                                  )
                fea = tf.image.resize_bilinear(fea,size=size,align_corners=True)
                branch_logits.append(fea)
                
            with tf.variable_scope("aspp0"):
                fea = features[-1]
                fea = slim.conv2d(fea,conv_dim,[1,1],
                                  activation_fn=self.activation_fn,
                                  normalizer_fn=self.normalizer_fn,
                                  normalizer_params=self.norm_params,
                                  biases_initializer=None if self.normalizer_fn is not None else tf.zeros_initializer(),
                                  )
                branch_logits.append(fea)

            for i,rate in enumerate(self.cfg.ATROUS_RATES,1):
                scope=f"aspp{i}"
                if self.cfg.ASPP_WITH_SEPARABLE_CONV:
                    aspp_fea = self.split_separable_conv2d(features[1],
                                                           filters=conv_dim,
                                                           rate=rate,
                                                           weight_decay=weight_decay,
                                                           scope=scope
                                                           )
                else:
                    aspp_fea = slim.conv2d(
                        features, conv_dim, 3, rate=rate, scope=scope,
                        activation_fn=self.activation_fn,
                        normalizer_fn=self.normalizer_fn,
                        normalizer_params=self.norm_params,
                    )
                
                branch_logits.append(aspp_fea)

            concat_logits = tf.concat(branch_logits, 3)
            concat_logits = slim.conv2d(
                concat_logits, conv_dim, 1,
                activation_fn=self.activation_fn,
                normalizer_fn=self.normalizer_fn,
                normalizer_params=self.norm_params,
                scope="concat_projection")
            if self.cfg.USE_DROP_BLOCKS and self.is_training:
                concat_logits = slim.dropout(
                    concat_logits,
                    keep_prob=0.9,
                    is_training=self.is_training,
                    scope="concat_projection"+ '_dropout')
            
            fea = self.refine_by_decoder(features[:-1],
                                         concat_logits,
                                         decoder_use_separable_conv=self.cfg.ASPP_WITH_SEPARABLE_CONV,
                                         weight_decay=weight_decay,
                                         reuse=reuse)
            
            if self.cfg.PRED_BACKGROUND:
                filters = self.cfg.NUM_CLASSES + 1
            else:
                filters = self.cfg.NUM_CLASSES
                
            logits = slim.conv2d(fea,filters,
                              kernel_size=1,
                              activation_fn=None,
                              normalizer_fn=None,
                              scope="semantic")
            return logits
            


    def refine_by_decoder(self,
                          low_features,
                          high_features,
                          decoder_use_separable_conv=False,
                          weight_decay=0.0001,
                          reuse=None):
        """Adds the decoder to obtain sharper segmentation results.

        Args:
          decoder_use_separable_conv: Employ separable convolution for decoder or not.
          weight_decay: The weight decay for model variables.
          reuse: Reuse the model variables or not.

        Returns:
          Decoder output with size [batch, decoder_height, decoder_width,
            decoder_channels].

        Raises:
          ValueError: If crop_size is None.
        """
        low_fea_conv_dim = self.cfg.LOW_FEA_CONV_DIM #48
        with slim.arg_scope(
                [slim.conv2d, slim.separable_conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                activation_fn=self.activation_fn,
                normalizer_fn=self.normalizer_fn,
                normalizer_params=self.norm_params,
                padding='SAME',
                stride=1,
                reuse=reuse):
            target_size = btf.img_size(low_features[0])
            decoder_features_list = [tf.image.resize_bilinear(high_features,target_size,align_corners=True)]
            with tf.variable_scope("decoder", "decoder", [high_features]):
                scope_suffix = ''
                for i,fea in enumerate(low_features):
                    fea = slim.conv2d(
                          fea,
                          low_fea_conv_dim,
                          1,
                          scope='feature_projection' + str(i) + scope_suffix)
                    if i != 0:
                        fea = tf.image.resize_bilinear(fea,target_size,align_corners=True)
                    decoder_features_list.append(fea)
                    
                decoder_depth = 256
                if decoder_use_separable_conv:
                    decoder_features = self.split_separable_conv2d(
                        tf.concat(decoder_features_list, 3),
                        filters=decoder_depth,
                        rate=1,
                        weight_decay=weight_decay,
                        scope='decoder_conv0' + scope_suffix)
                    decoder_features = self.split_separable_conv2d(
                        decoder_features,
                        filters=decoder_depth,
                        rate=1,
                        weight_decay=weight_decay,
                        scope='decoder_conv1' + scope_suffix)
                else:
                    num_convs = 2
                    decoder_features = slim.repeat(
                        tf.concat(decoder_features_list, 3),
                        num_convs,
                        slim.conv2d,
                        decoder_depth,
                        3,
                        scope='decoder_conv' + str(i) + scope_suffix)
            return decoder_features






