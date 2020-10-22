import numpy as np
import tensorflow as tf
from GN.graph import DynamicAdjacentMatrix,DynamicAdjacentMatrixShallow
import image_visualization as imv
import wsummary
import object_detection.bboxes as odb
from wtfop.wtfop_ops import adjacent_matrix_generator_by_iouv4
import wnnlayer as wnnl
import wml_tfutils as wmlt
import basic_tftools as btf
from functools import partial
from .abstractbbdnetx4 import AbstractBBDNet
import nlp.wlayers as nlpl
from wmodule import WModule
from object_detection2.standard_names import *
import object_detection2.od_toolkit as odt
import wtfop.wtfop_ops as wop
import object_detection2.wlayers as odl
from object_detection2.modeling.matcher import Matcher
import functools
import math
from object_detection2.data.dataloader import DataLoader
from .build import BBDNET_MODEL
import nlp.wlayers as nl


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
        self.mid_nms_outputs= []
        self.rnn_nr = rnn_nr
        self.conv_normalizer_fn,self.conv_normalizer_params = odt.get_norm('evo_norm_s0',is_training)
        self.conv_activation_fn = None
        self.normalizer_fn,self.normalizer_params = wnnl.graph_norm,{}
        self.activation_fn = tf.nn.leaky_relu
        self.preprocess_nr = preprocess_nr
        self.raw_bboxes = raw_bboxes
        self.input_labels = labels
        self.weight_decay = 1e-4
        self.POINT_HIDDEN_SIZE,self.EDGE_HIDDEN_SIZE,self.GLOBAL_HIDDEN_SIZE = self.cfg.MODEL.BBDNET.DIMS

        with tf.variable_scope("BBDNet",reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.fully_connected],
                                normalizer_fn=None,
                                normalizer_params=None,
                                activation_fn=self.activation_fn):
                self.build_net()

    def build_net(self):
        adj_mt = adjacent_matrix_generator_by_iouv4(bboxes=self.boxes, labels=self.input_labels,threshold=0.3, keep_connect=False)
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
            chl = btf.channel(self.map_data)
            map_data = tf.reduce_mean(self.map_data,axis=[1,2],keepdims=False)
            self.map_data = slim.fully_connected(map_data,chl)

        with tf.variable_scope("NodeEncode"):
            # node encode
            cxywh_boxes = odb.to_cxyhw(self.boxes)
            #cxywh_boxes = tf.zeros_like(cxywh_boxes)
            pos_data = tf.concat([self.boxes, cxywh_boxes], axis=1)
            net0 = slim.fully_connected(pos_data, self.POINT_HIDDEN_SIZE // 2,weights_regularizer=slim.l2_regularizer(self.weight_decay))

            # Process map data
            net2_0 = slim.fully_connected(self.map_data, self.POINT_HIDDEN_SIZE // 2)
            # process probability
            net2_1 = slim.fully_connected(self.probability, self.POINT_HIDDEN_SIZE // 2)
            net2 = tf.concat([net2_0, net2_1], axis=-1,name="concat_map_data_procs")
            # Fusion all parts of node.
            net = tf.concat([net0, net2], axis=1,name="concat_net0_net2")
            #net = tf.zeros_like(net)
            points_data = self._mlp(net, dims=self.POINT_HIDDEN_SIZE, scope="MLP_a")

        unit_nr = self.cfg.MODEL.BBDNET.RES_UNIT_NR
        edge_fn = partial(self.res_block, dims=self.EDGE_HIDDEN_SIZE, unit_nr=unit_nr, scope="UpdateEdge")
        point_fn = partial(self.res_block, dims=self.POINT_HIDDEN_SIZE, unit_nr=unit_nr, scope="UpdatePoint")


        edge_fn_i = partial(slim.fully_connected, num_outputs=self.EDGE_HIDDEN_SIZE, scope="UpdateEdge")
        #point_fn_i = partial(slim.fully_connected, num_outputs=self.POINT_HIDDEN_SIZE, scope="UpdatePoint")
        def point_fn_i(x):
            x = slim.fully_connected(x, num_outputs=self.POINT_HIDDEN_SIZE, scope="UpdatePoint")
            gamma = tf.get_variable("gamma", [1], initializer=tf.zeros_initializer())
            x0 = tf.expand_dims(x,axis=0)
            x0 = nl.self_attenation(x0,n_head=4)
            x0 = tf.squeeze(x0,axis=0)
            return x+gamma*x0

        if self.cfg.MODEL.BBDNET.USE_GLOBAL_ATTR:
            global_fn = partial(self.res_block, dims=self.POINT_HIDDEN_SIZE, unit_nr=unit_nr, scope="UpdateGlobal")
            global_fn_i = partial(slim.fully_connected,num_outputs=self.POINT_HIDDEN_SIZE, scope="UpdateGlobal")
        else:
            global_fn = None
            global_fn_i = None
        if self.cfg.MODEL.BBDNET.SHALLOW_GRAPH:
            grapy_t = DynamicAdjacentMatrixShallow
        else:
            grapy_t = DynamicAdjacentMatrix

        self.A = grapy_t(adj_mt=adj_mt,
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
                                    padding="VALID",
                                    weights_regularizer=slim.l2_regularizer(self.weight_decay))
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
            points_data1 = net2
            iou = odb.batch_bboxes_jaccard(senders_bboxes, receivers_bboxes)
            iou = tf.expand_dims(iou, axis=-1)
            iou = slim.fully_connected(iou,self.EDGE_HIDDEN_SIZE // 2,scope="encode_iou",
                                           weights_regularizer=slim.l2_regularizer(self.weight_decay))

            e_data0_s = tf.gather(points_data0, senders_indexs)
            e_data0_r = tf.gather(points_data0, receivers_indexs)
            e_data0 = e_data0_r - e_data0_s
            e_data0 = slim.fully_connected(e_data0,self.EDGE_HIDDEN_SIZE // 2,scope="MLP_a",
                                           weights_regularizer=slim.l2_regularizer(self.weight_decay))
            e_data1_s = tf.gather(points_data1, senders_indexs)
            e_data1_r = tf.gather(points_data1, receivers_indexs)
            e_data1 = (e_data1_r + e_data1_s) / 2.0
            e_data1 = slim.fully_connected(e_data1,self.EDGE_HIDDEN_SIZE // 2,scope="MLP_b",
                                           weights_regularizer=slim.l2_regularizer(self.weight_decay))
            e_data = tf.concat([e_data0, e_data1, iou], axis=1)
            e_data = slim.fully_connected(e_data, self.EDGE_HIDDEN_SIZE, scope="EdgeEncode",
                                          weights_regularizer=slim.l2_regularizer(self.weight_decay))
            #e_data = tf.zeros_like(e_data)
            self.A.edges_data = e_data

        latent0 = {"edges": e_data, "nodes": points_data, "global": self.A.global_attr}
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        def bboxes_output_fn(net):
            with tf.variable_scope("NodesBBoxesOutput", reuse=tf.AUTO_REUSE):
                # net = AbstractBBDNet.max_pool(net)
                net = slim.fully_connected(net,  self.POINT_HIDDEN_SIZE)
                net = slim.fully_connected(net, 4,
                                           normalizer_fn=None,
                                           activation_fn=None)
                return net

        def nms_output_fn(net):
            with tf.variable_scope("NodesNMSOutput", reuse=tf.AUTO_REUSE):
                # net = AbstractBBDNet.max_pool(net)
                net = slim.fully_connected(net,  self.POINT_HIDDEN_SIZE)
                net = slim.fully_connected(net, 1,
                                           normalizer_fn=None,
                                           activation_fn=None)
                net = tf.squeeze(net,axis=-1)
                return net

        def edge_output_fn(net):
            with tf.variable_scope("EdgesOutput", reuse=tf.AUTO_REUSE):
                # net = AbstractBBDNet.max_pool(net)
                net = slim.fully_connected(net,  self.EDGE_HIDDEN_SIZE)
                net = slim.fully_connected(net, 1,
                                           biases_initializer=tf.constant_initializer(value=bias_value),
                                           normalizer_fn=None,
                                           activation_fn=None)
                net = tf.squeeze(net,axis=-1)
                return net

        for i in range(self.preprocess_nr):
            with tf.variable_scope(f"Layer_{i}"):
                if i > 0:
                    self.A.concat(latent0)
                self.A.update_independent(point_fn_i,edge_fn_i,global_fn_i,["UpdatePoint_i","UpdateEdge_i","UpdateGlobal_i"])
                self.A.update(point_fn, edge_fn, global_fn, ["UpdatePoint", "UpdateEdge", "UpdateGlobal"])


        print("Rnn nr:", self.rnn_nr)
        for i in range(self.rnn_nr):
            with tf.variable_scope("LayerRNN"):
                self.A.concat(latent0,use_global_attr=False)
                self.A.update_independent(point_fn_i,edge_fn_i,None,["UpdatePoint_i","UpdateEdge_i","UpdateGlobal_i"])
                self.A.update(point_fn, edge_fn, None, ["UpdatePoint", "UpdateEdge", "UpdateGlobal"],
                              use_global_attr=False)
            if i>0 or self.rnn_nr==1:
                self.mid_bboxes_outputs.append(bboxes_output_fn(self.A.points_data))
                self.mid_nms_outputs.append(nms_output_fn(self.A.points_data))
                if edge_output_fn is not None:
                    self.mid_edges_outputs.append(edge_output_fn(self.A.edges_data))
        self.pred_bboxes_deltas = self.mid_bboxes_outputs[-1]
        self.nms_logits = self.mid_nms_outputs[-1]
        return self.logits

    '''
    y:[batch_size,k] target label
    '''
    def loss(self, nms_y,indices,bboxes,gboxes,glabels,glens):
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

        with tf.name_scope("nms_loss"):
            scale = 200.0
            for i, logits in enumerate(self.mid_nms_outputs):
                nms_loss = self._lossv2(logits, nms_y,log=False)
                wmlt.variable_summaries_v2(nms_loss, f"nms_loss_{i}")
                loss_list.append(nms_loss * scale)

        if self.cfg.MODEL.BBDNET.USE_EDGE_LOSS:
            with tf.name_scope("edges_loss"):
                senders_indexs, receivers_indexs = self.A.senders_indexs, self.A.receivers_indexs
                edge_y = tf.gather(nms_y, receivers_indexs)
                scale = 0.05
                for i, logits in enumerate(self.mid_edges_outputs):
                    e_loss = self._lossv2(logits, edge_y)
                    wmlt.variable_summaries_v2(e_loss, f"e_loss_{i}")
                    loss_list.append(e_loss * scale)

        return tf.add_n(loss_list)

@BBDNET_MODEL.register()
class BBDNetX4(WModule):
    def __init__(self,num_classes,max_node_nr=96,cfg=None,parent=None,*args,**kwargs):
        super().__init__(cfg=cfg,parent=parent,*args,**kwargs)
        self.num_classes = num_classes
        self.threshold = [self.cfg.MODEL.BBDNET.SCORE_THRESH_TEST,0.3]
        self.max_node_nr = max_node_nr
        self.conv_normalizer_fn,self.conv_normalizer_params = odt.get_norm('evo_norm_s0',True)
        self.conv_activation_fn = None
        self.matcher = Matcher(thresholds=[0.5],cfg=self.cfg, parent=self)

    def forward(self,datas):
        datas = dict(datas)
        loss = {}
        '''with tf.variable_scope("BBDNet"):
            map_data = tf.stop_gradient(datas['tower_nets'])
            map_data = slim.conv2d(map_data, 256, [1,1], normalizer_fn=self.conv_normalizer_fn,
                                   normalizer_params=self.conv_normalizer_params,
                                   activation_fn=self.conv_activation_fn,
                                   scope="smooth_tower_nets")
            datas['net_data'] = map_data
            datas.pop('tower_nets')'''

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
        #probs = tf.stop_gradient(datas[RD_RAW_PROBABILITY][:l])
        labels = tf.stop_gradient(datas[RD_LABELS][:l])
        raw_probs = datas[RD_PROBABILITY][:l]
        probs = tf.expand_dims(raw_probs,axis=-1)
        #probs = tf.zeros_like(probs)
        return_nr = tf.shape(datas[RD_LABELS])[0]
        base_net = tf.stop_gradient(datas['base_net'])
        #base_net = tf.zeros_like(base_net)
        net_data = datas['net_data']
        if len(net_data.get_shape()) == 3:
            pooler = odl.WROIAlign(bin_size=[1,1],output_size=[7,7])
            crop_bboxes = odb.scale_bboxes(raw_bboxes,0.1)
            map_data = pooler(tf.expand_dims(net_data,axis=0),tf.expand_dims(crop_bboxes,axis=0))
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
        raw_labels = labels
        if self.is_training:
            #for box regression loss
            y, y_scores, indexs = self.matcher(boxes=tf.expand_dims(raw_bboxes, axis=0),
                                               gboxes=tf.expand_dims(datas[GT_BOXES], axis=0),
                                               glabels=tf.expand_dims(datas[GT_LABELS], axis=0),
                                               glength=tf.reshape(datas[GT_LENGTH], [1]))
            nms_y, nms_y_scores,nms_indexs = wop.boxes_match_with_predv3(boxes=tf.expand_dims(raw_bboxes,axis=0),
                                                    plabels=tf.expand_dims(raw_labels, axis=0),
                                                    pprobs=tf.expand_dims(raw_probs, axis=0),
                                                    gboxes=tf.expand_dims(datas[GT_BOXES],axis=0),
                                                    glabels=tf.expand_dims(datas[GT_LABELS],axis=0),
                                                    glens=tf.reshape(datas[GT_LENGTH],[1]),
                                                    sort_by_probs=False,
                                                    threshold=0.5)
            y = tf.squeeze(y,axis=0)
            nms_y = tf.squeeze(nms_y,axis=0)
            nms_y = tf.cast(tf.greater(nms_y, 0), tf.int32)
            m_bboxes = tf.boolean_mask(raw_bboxes, y > 0)
            m_nms_bboxes = tf.boolean_mask(raw_bboxes, nms_y > 0)
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
            wsummary.detection_image_summary(tf.expand_dims(datas[IMAGE],axis=0),
                                             tf.expand_dims(m_nms_bboxes, axis=0),
                                             classes=tf.expand_dims(tf.boolean_mask(nms_y, nms_y > 0),axis=0),
                                             category_index=DataLoader.category_index,
                                             name='match_nms_boxes')
            outputs['bbd_net_loss'] = bbd_net.loss(nms_y,
                                                   indices=tf.squeeze(indexs,axis=0),
                                                   bboxes=raw_bboxes,
                                                   gboxes=datas[GT_BOXES],
                                                   glabels=datas[GT_LABELS],
                                                   glens=tf.reshape(datas[GT_LENGTH],[1]))

        fboxes, flabels, probs, raw_plabels = bbd_net.get_predict(raw_bboxes,raw_labels,raw_probs,threshold=self.threshold)

        r_l = tf.shape(flabels)[0]
        pad_nr = return_nr-r_l
        outputs[RD_BOXES] = tf.pad(fboxes,paddings=[[0,pad_nr],[0,0]])
        outputs[RD_LABELS] = tf.pad(flabels,paddings=[[0,pad_nr]])
        outputs[RD_PROBABILITY] = tf.pad(probs,paddings=[[0,pad_nr]])
        outputs[RD_LENGTH] = tf.shape(flabels)[0]

        return outputs
