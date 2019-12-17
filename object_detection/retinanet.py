#coding=utf-8
import tensorflow as tf
from object_detection.ssd import SSD
slim = tf.contrib.slim

class RetinaNet(SSD):
    def __init__(self,num_classes,batch_size=1,is_training=False):
        super().__init__(num_classes,batch_size=batch_size)
        self.is_training = is_training

    def _buildNet(self,bimage,reuse):
        pass
        '''
        with slim.arg_scope(resnet_arg_scope()):
            with tf.variable_scope("FeatureExtractor"):
                net,image_features = resnet_v1_50_with_block4(bimage,output_stride=None,is_training=self.is_training,store_non_strided_activations=True)
                with slim.arg_scope([slim.batch_norm], is_training=self.is_training):
                    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu6):
                        with tf.variable_scope("resnet_v1_50/fpn",reuse=False):
                            base_fpn_max_level = min(7, 5)
                            feature_block_list = []
                            for level in range(3, 6):
                                feature_block_list.append('block{}'.format(level - 1))
                            fpn_features = self.fpn_top_down_feature_maps(
                                [(key, image_features['FeatureExtractor/resnet_v1_50/'+key]) for key in feature_block_list],
                                depth=256)
                            feature_maps = []
                            for level in range(3, base_fpn_max_level + 1):
                                feature_maps.append(
                                    fpn_features['top_down_block{}'.format(level - 1)])
                            last_feature_map = fpn_features['top_down_block{}'.format(
                                base_fpn_max_level - 1)]
                            for i in range(base_fpn_max_level, 7):
                                last_feature_map = slim.conv2d(
                                    last_feature_map,
                                    num_outputs=256,
                                    kernel_size=[3, 3],
                                    stride=2,
                                    padding='SAME',
                                    scope='bottom_up_block{}'.format(i))
                                feature_maps.append(last_feature_map)
            self.getAnchorBoxesByFreaturesV4(feature_maps,min_level=3,max_level=7,
                                                      aspect_ratios=self.ratios,
                                                      anchor_scale=4.0,
                                                      scales_per_octave=2,size=bimage.get_shape().as_list()[1:3])
            self.buildPredictor(feature_maps=feature_maps)
        return feature_maps'''
    '''
    feature_maps:shape=[[h,w,c0],h/2,w/2,c1],....]
    '''
    def buildPredictor(self, feature_maps,num_layers_before_predictor=4,activation_fn=tf.nn.relu6):
        logits_list = []
        boxes_regs_list = []
        with slim.arg_scope([slim.batch_norm], is_training=self.is_training):
            with slim.arg_scope([slim.conv2d], activation_fn=None):
                net_data = list(enumerate(feature_maps))
                for i, net in net_data:
                    channel = net.get_shape().as_list()[-1]
                    with tf.variable_scope("WeightSharedConvolutionalBoxPredictor", reuse=tf.AUTO_REUSE):
                        logits_nr = self.logits_nr_list[i]
                        if self.pred_bboxes_classwise:
                            regs_nr = logits_nr * (self.num_classes - 1)
                        else:
                            regs_nr = logits_nr
                        with tf.variable_scope("ClassPredictionTower"):
                            net0 = net
                            for j in range(num_layers_before_predictor):
                                net0 = slim.conv2d(net0, channel, [3, 3], scope=f"conv2d_{j}", normalizer_fn=None,
                                                   biases_initializer=None,padding="SAME")
                                net0 = slim.batch_norm(net0, scope=f'conv2d_{j}/BatchNorm/feature_{i}')
                                net0 = activation_fn(net0)
                        logits_net = slim.conv2d(net0, logits_nr * self.num_classes, [3, 3], activation_fn=None,
                                                 normalizer_fn=None,
                                                 scope="ClassPredictor")
                        with tf.variable_scope("BoxPredictionTower"):
                            net1 = net
                            for j in range(num_layers_before_predictor):
                                net1 = slim.conv2d(net1, channel, [3, 3], scope=f"conv2d_{j}", normalizer_fn=None,
                                                   biases_initializer=None,padding="SAME")
                                net1 = slim.batch_norm(net1, scope=f'conv2d_{j}/BatchNorm/feature_{i}')
                                net1 = activation_fn(net1)
                        boxes_regs = slim.conv2d(net1, regs_nr * 4, [3, 3], activation_fn=None,
                                                 normalizer_fn=None,
                                                 scope="BoxPredictor")

                        logits_list.append(logits_net)
                        boxes_regs_list.append(boxes_regs)


        logits = self.merge_classes_predictor(logits_list, self.num_classes)
        self.regs = self.merge_box_predictor(boxes_regs_list)
        self.logits = logits
