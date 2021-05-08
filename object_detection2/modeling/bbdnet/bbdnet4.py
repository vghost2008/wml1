import numpy as np
import tensorflow as tf
from GN.graph import DynamicAdjacentMatrix
import image_visualization as imv
import wsummary
import object_detection2.bboxes as odb
from wtfop.wtfop_ops import adjacent_matrix_generator_by_iou,adjacent_matrix_generator_by_iouv2
import wnnlayer as wnnl
import wml_tfutils as wmlt
import basic_tftools as btf
from functools import partial
from .abstractbbdnet4 import AbstractBBDNet
import nlp.wlayers as nlpl
from wmodule import WModule
from object_detection2.standard_names import *
import object_detection2.od_toolkit as odt
import wtfop.wtfop_ops as wop
import object_detection2.wlayers as odl
import functools
import math
from object_detection2.data.dataloader import DataLoader
from object_detection2.modeling.matcher import Matcher
from .build import BBDNET_MODEL


slim = tf.contrib.slim


'''
Each time only process one example.
'''
class BBDNetForOneImg(AbstractBBDNet):
    '''
    boxes: the  boxes, [batch_size=1,k,4]
    probability: [batch_size=1,k,classes_num] the probability of boxes
    map_data:[batch_size=1,k,C]
    classes_num: ...
    '''

    def __init__(self, cfg,boxes, probability, labels,map_data, classes_num, base_net, raw_bboxes=None,is_training=False, rnn_nr=3,preprocess_nr=2):
        super().__init__(cfg,boxes, probability, map_data, classes_num, base_net, is_training)
        self.mid_edges_outputs = []
        self.mid_global_outputs = []
        self.rnn_nr = rnn_nr
        self.conv_normalizer_fn,self.conv_normalizer_params = odt.get_norm('evo_norm_s0',is_training)
        self.conv_activation_fn = None
        self.normalizer_fn,self.normalizer_params = wnnl.graph_norm,{}
        self.activation_fn = tf.nn.leaky_relu
        self.preprocess_nr = preprocess_nr
        self.raw_bboxes = raw_bboxes
        self.input_labels = labels

        self.POINT_HIDDEN_SIZE,self.EDGE_HIDDEN_SIZE,self.GLOBAL_HIDDEN_SIZE = self.cfg.MODEL.BBDNET.DIMS

        with tf.variable_scope("BBDNet",reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.fully_connected],
                                normalizer_fn=None,
                                normalizer_params=None,
                                activation_fn=self.activation_fn):
                self.build_net()

    def build_net(self):
        adj_mt = adjacent_matrix_generator_by_iou(bboxes=self.boxes, threshold=0.3, keep_connect=False)
        '''adj_mt = adjacent_matrix_generator_by_iouv2(bboxes=self.boxes,
                                                  labels=self.input_labels,
                                                  probs=self.probability,
                                                  threshold=[0.3,0.12],
                                                  keep_connect=False)'''
        '''shape = tf.shape(self.boxes)
        adj_mt = tf.ones(shape=[shape[0],shape[0]],dtype=tf.int32)
        adj_mt = adj_mt - tf.eye(num_rows=shape[0],dtype=tf.int32)'''
        #adj_mt = tf.Print(adj_mt,["adj_mt_shape:",tf.shape(adj_mt)])
        self.adj_mt = adj_mt
        if len(self.map_data.get_shape()) == 4:
            with tf.variable_scope("smooth_map_data"):
                chl = btf.channel(self.map_data)
                map_data = slim.conv2d(self.map_data,chl,[3,3],normalizer_fn=self.conv_normalizer_fn,
                                       normalizer_params=self.conv_normalizer_params,
                                       activation_fn=self.conv_activation_fn)
                map_data = slim.conv2d(map_data,chl,[3,3],normalizer_fn=self.conv_normalizer_fn,
                                       normalizer_params=self.conv_normalizer_params,
                                       activation_fn=self.conv_activation_fn)
                map_data = slim.conv2d(map_data,chl,[3,3],normalizer_fn=self.conv_normalizer_fn,
                                       normalizer_params=self.conv_normalizer_params,
                                       activation_fn=self.conv_activation_fn)
                map_data = tf.reduce_mean(map_data,axis=[1,2],keepdims=False)
                self.map_data = slim.fully_connected(map_data,chl)

        with tf.variable_scope("NodeEncode"):
            # node encode
            cxywh_boxes = odb.to_cxyhw(self.boxes)
            #cxywh_boxes = tf.zeros_like(cxywh_boxes)
            pos_data = tf.concat([self.boxes, cxywh_boxes], axis=1)
            pos_data = slim.fully_connected(pos_data, 32, )
            raw_points = tf.concat([pos_data, self.probability], axis=1)
            # generatic point hide attribute
            net0 = slim.fully_connected(raw_points, self.POINT_HIDDEN_SIZE // 2)

            # Process map data
            net2 = BBDNetForOneImg.self_attenation(tf.expand_dims(self.map_data, axis=0), n_head=2,
                                              is_training=self.is_training, normalizer_fn=self.normalizer_fn)
            net2 = tf.squeeze(net2, axis=0)
            net2 = slim.fully_connected(net2, self.POINT_HIDDEN_SIZE // 2)
            # Fusion all parts of node.
            net = tf.concat([net0, net2], axis=1)
            #net = tf.zeros_like(net)
            points_data = self._mlp(net, dims=self.POINT_HIDDEN_SIZE, scope="MLP_a")

        unit_nr = self.cfg.MODEL.BBDNET.RES_UNIT_NR
        edge_fn = partial(self.res_block, dims=self.EDGE_HIDDEN_SIZE, unit_nr=unit_nr, scope="UpdateEdge")
        point_fn = partial(self.res_block, dims=self.POINT_HIDDEN_SIZE, unit_nr=unit_nr, scope="UpdatePoint")
        edge_fn_i = partial(slim.fully_connected, num_outputs=self.EDGE_HIDDEN_SIZE, scope="UpdateEdge")
        point_fn_i = partial(slim.fully_connected, num_outputs=self.POINT_HIDDEN_SIZE, scope="UpdatePoint")
        if self.cfg.MODEL.BBDNET.USE_GLOBAL_ATTR:
            global_fn = partial(self.res_block, dims=self.POINT_HIDDEN_SIZE, unit_nr=unit_nr, scope="UpdateGlobal")
            global_fn_i = partial(slim.fully_connected,num_outputs=self.POINT_HIDDEN_SIZE, scope="UpdateGlobal")
        else:
            global_fn = None
            global_fn_i = None

        self.A = DynamicAdjacentMatrix(adj_mt=adj_mt,
                                       points_data=points_data,
                                       edges_data=None,
                                       edges_data_dim=self.EDGE_HIDDEN_SIZE)
        self.A.use_sent_edges_for_node = self.cfg.MODEL.BBDNET.USE_SENT_EDGES_FOR_NODE
        self.A.redges_reducer_for_points = functools.partial(tf.reduce_sum,axis=0,keepdims=False)
        self.A.sedges_reducer_for_points = functools.partial(tf.reduce_sum,axis=0,keepdims=False)

        if self.cfg.MODEL.BBDNET.EDGES_REDUCER_FOR_POINTS == "sum":
            print("sum reducer")
            self.A.edges_reducer_for_points = tf.unsorted_segment_sum
        elif self.cfg.MODEL.BBDNET.EDGES_REDUCER_FOR_POINTS == "mean":
            print("mean reducer")
            self.A.edges_reducer_for_points = tf.unsorted_segment_mean

        # global encode
        if self.cfg.MODEL.BBDNET.USE_GLOBAL_ATTR:
            with tf.variable_scope("InitGlobalAttr"):
                gattr = slim.conv2d(tf.expand_dims(self.base_net,axis=0),
                                              self.GLOBAL_HIDDEN_SIZE, [3, 3],
                                    normalizer_fn=self.conv_normalizer_fn,
                                    normalizer_params=self.conv_normalizer_params,
                                    activation_fn=self.conv_activation_fn,
                                    padding="VALID")
                gattr = tf.reduce_mean(gattr, axis=[1, 2], keepdims=False)
                self.A.global_attr = gattr
                self.A.update_global_independent(global_fn_i)
        else:
            self.A.global_attr = None

        # edge encode
        with tf.variable_scope("EdgeEncode"):
            senders_indexs, receivers_indexs = self.A.senders_indexs, self.A.receivers_indexs
            senders_bboxes = tf.gather(self.boxes, senders_indexs)
            receivers_bboxes = tf.gather(self.boxes, receivers_indexs)
            points_data0 = tf.concat([self.boxes, cxywh_boxes[:, :2]], axis=1)
            points_data1 = tf.concat([net2, self.probability], axis=1)
            iou = odb.batch_bboxes_jaccard(senders_bboxes, receivers_bboxes)
            iou = tf.expand_dims(iou, axis=-1) * tf.ones(shape=[1, self.EDGE_HIDDEN_SIZE // 2])

            e_data0_s = tf.gather(points_data0, senders_indexs)
            e_data0_r = tf.gather(points_data0, receivers_indexs)
            e_data0 = e_data0_r - e_data0_s
            e_data0 = slim.fully_connected(e_data0,self.EDGE_HIDDEN_SIZE // 2,scope="MLP_a")
            e_data1_s = tf.gather(points_data1, senders_indexs)
            e_data1_r = tf.gather(points_data1, receivers_indexs)
            e_data1 = (e_data1_r + e_data1_s) / 2.0
            e_data1 = slim.fully_connected(e_data1,self.EDGE_HIDDEN_SIZE // 2,scope="MLP_b")
            e_data = tf.concat([e_data0, e_data1, iou], axis=1)
            e_data = slim.fully_connected(e_data, self.EDGE_HIDDEN_SIZE, scope="EdgeEncode")
            #e_data = tf.zeros_like(e_data)
            self.A.edges_data = e_data

        latent0 = {"edges": e_data, "nodes": points_data, "global": self.A.global_attr}
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        def output_fn(net):
            with tf.variable_scope("NodesOutput", reuse=tf.AUTO_REUSE):
                # net = AbstractBBDNet.max_pool(net)
                net = slim.fully_connected(net,  self.POINT_HIDDEN_SIZE)
                net = slim.fully_connected(net, self.classes_num+1,
                                           biases_initializer=tf.constant_initializer(value=bias_value),
                                           normalizer_fn=None,
                                           activation_fn=None)
                return net

        def bboxes_output_fn(net):
            with tf.variable_scope("NodesBBoxesOutput", reuse=tf.AUTO_REUSE):
                # net = AbstractBBDNet.max_pool(net)
                net = slim.fully_connected(net,  self.POINT_HIDDEN_SIZE)
                net = slim.fully_connected(net, 4,
                                           normalizer_fn=None,
                                           activation_fn=None)
                return net

        def edge_output_fn(net):
            with tf.variable_scope("EdgesOutput", reuse=tf.AUTO_REUSE):
                # net = AbstractBBDNet.max_pool(net)
                net = slim.fully_connected(net,  self.EDGE_HIDDEN_SIZE)
                net = slim.fully_connected(net, 1,
                                           biases_initializer=tf.constant_initializer(value=bias_value),
                                           normalizer_fn=None,
                                           activation_fn=None)
                return net
        if self.cfg.MODEL.BBDNET.USE_GLOBAL_ATTR and self.cfg.MODEL.BBDNET.USE_GLOBAL_LOSS:
            def global_output_fn(net):
                with tf.variable_scope("GlobalOutput", reuse=tf.AUTO_REUSE):
                    # net = AbstractBBDNet.max_pool(net)
                    net = slim.fully_connected(net, max(self.GLOBAL_HIDDEN_SIZE // 2,self.classes_num+1))
                    net = slim.fully_connected(net, self.classes_num,
                                               normalizer_fn=None,
                                               activation_fn=None)
                    net = tf.reshape(net,[-1])
                    return net
        else:
            global_output_fn = None

        for i in range(self.preprocess_nr):
            with tf.variable_scope(f"Layer_{i}"):
                if i > 0:
                    self.A.concat(latent0)
                self.A.update_independent(point_fn_i,edge_fn_i,global_fn_i,["UpdatePoint_i","UpdateEdge_i","UpdateGlobal_i"])
                self.A.update(point_fn, edge_fn, global_fn, ["UpdatePoint", "UpdateEdge", "UpdateGlobal"])


        print("Rnn nr:", self.rnn_nr)
        for i in range(self.rnn_nr):
            with tf.variable_scope("LayerRNN"):
                self.A.concat(latent0,use_global_attr=True)
                self.A.update_independent(point_fn_i,edge_fn_i,global_fn_i,["UpdatePoint_i","UpdateEdge_i","UpdateGlobal_i"])
                self.A.update(point_fn, edge_fn, global_fn, ["UpdatePoint", "UpdateEdge", "UpdateGlobal"],
                              use_global_attr=True)
            if i>0 or self.rnn_nr==1:
                self.mid_outputs.append(output_fn(self.A.points_data))
                self.mid_bboxes_outputs.append(bboxes_output_fn(self.A.points_data))
                if edge_output_fn is not None:
                    self.mid_edges_outputs.append(edge_output_fn(self.A.edges_data))
                if global_output_fn is not None:
                    self.mid_global_outputs.append(global_output_fn(self.A.global_attr))
        self.logits = self.mid_outputs[-1]
        self.pred_bboxes_deltas = self.mid_bboxes_outputs[-1]
        return self.logits

    '''
    y:[batch_size,k] target label
    '''
    def loss(self, y,indices,bboxes,gboxes,glabels,glens):
        assert y.get_shape().ndims == 1, "error"
        loss_list = []
        with tf.name_scope("bboxes_regression_loss"):
            foreground_idxs = tf.greater_equal(indices,0)
            gt_bboxes = tf.gather(gboxes,tf.nn.relu(indices))
            gt_bboxes = tf.boolean_mask(gt_bboxes, foreground_idxs)
            bboxes = tf.boolean_mask(bboxes, foreground_idxs)
            box_loss = []
            scale = 1.0
            with tf.variable_scope("losses"):
                for i, deltas in enumerate(self.mid_bboxes_outputs):
                    deltas = tf.boolean_mask(deltas, foreground_idxs)
                    box = self.box2box_transform.apply_deltas(deltas, bboxes)
                    reg_loss_sum = odl.giou_loss(box, gt_bboxes)
                    reg_loss_sum = tf.reduce_mean(reg_loss_sum)*scale
                    box_loss.append(reg_loss_sum)

            box_loss = tf.add_n(box_loss)

        wsummary.histogram_or_scalar(box_loss,"box_loss")
        loss_list.append(box_loss)

        with tf.name_scope("nodes_loss"):
            node_loss = super().loss(y)
            loss_list.append(node_loss)

        if self.cfg.MODEL.BBDNET.USE_EDGE_LOSS:
            with tf.name_scope("edges_loss"):
                senders_indexs, receivers_indexs = self.A.senders_indexs, self.A.receivers_indexs
                receivers_y = tf.gather(y, receivers_indexs)
                receivers_y = tf.cast(tf.greater(receivers_y, 0), tf.int32)
                edge_y = receivers_y
                scale = 0.05
                for i, logits in enumerate(self.mid_edges_outputs):
                    e_loss = self._lossv2(logits, edge_y)
                    wmlt.variable_summaries_v2(e_loss, f"e_loss_{i}")
                    loss_list.append(e_loss * scale)

        if self.cfg.MODEL.BBDNET.USE_GLOBAL_LOSS:
            scale = 1.0
            with tf.name_scope("global_loss"):
                data = tf.zeros(shape=[self.classes_num+1])
                gy = wop.set_value(tensor=data,v=tf.constant(1,dtype=tf.float32,shape=()),index=tf.reshape(y,[-1,1]))
                gy = gy[1:]
                for i, logits in enumerate(self.mid_global_outputs):
                    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=gy))*scale
                    loss_list.append(loss)

        return tf.add_n(loss_list)


    @staticmethod
    def self_attenation(net, n_head=1, keep_prob=None, is_training=False, scope=None,
                        normalizer_fn=wnnl.layer_norm, normalizer_params=None):
        with tf.variable_scope(scope, default_name="non_local"):
            shape = net.get_shape().as_list()
            channel = shape[-1]
            out = nlpl.self_attenation(net, n_head=n_head, keep_prob=keep_prob, is_training=is_training,
                                       use_mask=False)
            out = tf.layers.dense(out, channel,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            out = out + net
            if normalizer_params is None:
                normalizer_params = {}
            out = normalizer_fn(out, **normalizer_params)
            return out

@BBDNET_MODEL.register()
class BBDNet4(WModule):
    def __init__(self,num_classes,max_node_nr=96,cfg=None,parent=None,*args,**kwargs):
        super().__init__(cfg=cfg,parent=parent,*args,**kwargs)
        self.num_classes = num_classes
        self.threshold = self.cfg.MODEL.BBDNET.SCORE_THRESH_TEST
        self.max_node_nr = max_node_nr
        self.matcher = Matcher(thresholds=[0.5],cfg=self.cfg, parent=self)

    def forward(self,datas):
        datas = dict(datas)
        loss = {}
        if self.is_training:
            res = wmlt.static_or_dynamic_map_fn(self.process_one_image,elems=datas,
                        dtype={'bbd_net_loss': tf.float32, 'boxes': tf.float32, 'labels': tf.int32,
                               'probability': tf.float32, 'length': tf.int32})
            loss['bbd_net_loss'] = tf.reduce_mean(res.pop('bbd_net_loss'))
        else:
            res = wmlt.static_or_dynamic_map_fn(self.process_one_image,elems=datas,
                            dtype={'boxes': tf.float32, 'labels': tf.int32,
                                   'probability': tf.float32, 'length': tf.int32})

        return loss,res
    
    def process_one_image(self,datas):
        l = tf.minimum(datas[RD_LENGTH],self.max_node_nr)
        bboxes = tf.stop_gradient(datas[RD_BOXES][:l])
        raw_bboxes = bboxes
        if self.cfg.MODEL.BBDNET.ABSOLUTE_BBOXES:
            print("use absolute bboxes for bbdnet.")
            img_size = tf.to_float(tf.shape(datas[IMAGE]))
            bboxes = odb.tfrelative_boxes_to_absolutely_boxes(bboxes,width=img_size[1],height=img_size[0])
        #bboxes = tf.zeros_like(bboxes)
        probs = tf.stop_gradient(datas[RD_RAW_PROBABILITY][:l])
        labels = tf.stop_gradient(datas[RD_LABELS][:l])
        #probs = tf.zeros_like(probs)
        return_nr = tf.shape(datas[RD_LABELS])[0]
        base_net = tf.stop_gradient(datas['base_net'])
        #base_net = tf.zeros_like(base_net)
        net_data = datas['net_data']
        if len(net_data.get_shape()) == 2 and RD_INDICES in datas:
            map_data = tf.stop_gradient(tf.gather(datas['net_data'],datas[RD_INDICES][:l]))
        elif len(net_data.get_shape()) == 3:
            pooler = odl.WROIAlign(bin_size=[1,1],output_size=[7,7])
            net_data = pooler(tf.expand_dims(net_data,axis=0),tf.expand_dims(raw_bboxes,axis=0))
            map_data = tf.stop_gradient(net_data)
            print(f"Crop bbox attr for bbdnet.")
        else:
            raise NotImplementedError(f"Error net data input.")

        #map_data = tf.zeros_like(map_data)
        bbd_net = BBDNetForOneImg(self.cfg,
                                bboxes,
                                probs,
                                labels,
                                map_data,
                                self.num_classes,
                                base_net, 
                                self.is_training,
                                rnn_nr=self.cfg.MODEL.BBDNET.NUM_PROCESSING_STEPS,
                                preprocess_nr=self.cfg.MODEL.BBDNET.NUM_PREPROCESSING_STEPS)
        outputs = {}
        if self.is_training:
            y, y_scores,indexs = self.matcher(boxes=tf.expand_dims(raw_bboxes,axis=0),
                                         gboxes=tf.expand_dims(datas[GT_BOXES],axis=0),
                                         glabels=tf.expand_dims(datas[GT_LABELS],axis=0),
                                         glength=tf.reshape(datas[GT_LENGTH],[1]))
            y = tf.squeeze(y,axis=0)
            m_bboxes = tf.boolean_mask(raw_bboxes, y > 0)
            img = datas[IMAGE]
            img = imv.draw_graph_by_bboxes(img,raw_bboxes,bbd_net.adj_mt)
            tf.summary.image("img_with_graph",tf.expand_dims(img,axis=0))
            wsummary.detection_image_summary(tf.expand_dims(img,axis=0),
                                             tf.expand_dims(raw_bboxes, axis=0),
                                             classes=tf.expand_dims(labels,axis=0),
                                             category_index=DataLoader.category_index,
                                             name='img_boxes_graph',
                                             max_boxes_to_draw=100,
                                             min_score_thresh=0.001)
            wsummary.detection_image_summary(tf.expand_dims(datas[IMAGE],axis=0),
                                             tf.expand_dims(m_bboxes, axis=0),
                                             classes=tf.expand_dims(tf.boolean_mask(y, y > 0),axis=0),
                                             category_index=DataLoader.category_index,
                                             name='match_boxes')
            outputs['bbd_net_loss'] = bbd_net.loss(y,
                                                   indices=tf.squeeze(indexs,axis=0),
                                                   bboxes=raw_bboxes,
                                                   gboxes=datas[GT_BOXES],
                                                   glabels=datas[GT_LABELS],
                                                   glens=tf.reshape(datas[GT_LENGTH],[1]))

        fboxes, flabels, probs, raw_plabels = bbd_net.get_predict(raw_bboxes,threshold=self.threshold)

        r_l = tf.shape(flabels)[0]
        pad_nr = return_nr-r_l
        #pad_nr = tf.Print(pad_nr,["bbd_return_nr",l,r_l,datas[GT_LENGTH]])
        outputs[RD_BOXES] = tf.pad(fboxes,paddings=[[0,pad_nr],[0,0]])
        outputs[RD_LABELS] = tf.pad(flabels,paddings=[[0,pad_nr]])
        outputs[RD_PROBABILITY] = tf.pad(probs,paddings=[[0,pad_nr]])
        outputs[RD_LENGTH] = tf.shape(flabels)[0]

        return outputs
