#coding=utf-8
import tensorflow as tf
import math
import tfop
import wml_tfutils as wmlt
from object_detection2.datadef import EncodedData
import object_detection2.bboxes as odb
from basic_tftools import batch_size
import basic_tftools as btf

_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)
class AbstractBox2BoxTransform(object):
    def get_deltas(self,boxes,gboxes,labels,indices,img_size=None):
        """
        the labels,indices is the output of matcher
        boxes:[batch_size,N,4]
        gboxes:[batch_size,M,4]
        labels:[batch_size,N]
        indices:[batch_size,N]
        output:
        [batch_size,N,4]
        """
        pass

    def get_deltas_by_proposals_data(self,proposals:EncodedData,img_size=None):
        """
        the labels,indices is the output of matcher
        boxes:[batch_size,N,4]
        gboxes:[batch_size,M,4]
        labels:[batch_size,N]
        indices:[batch_size,N]
        output:
        [batch_size,N,4]
        """
        boxes = proposals.boxes
        gboxes = proposals.gt_boxes
        indices = proposals.indices
        labels = proposals.gt_object_logits
        return self.get_deltas(boxes=boxes,gboxes=gboxes,labels=labels,indices=indices,img_size=img_size)

    def apply_deltas(self,deltas,boxes,img_size=None):
        '''
        :param deltas: [batch_size,N,4]/[N,4]
        :param boxes: [batch_size,N,4]/[N.4]
        :return:
        '''
        pass

class Box2BoxTransform(AbstractBox2BoxTransform):
    '''
    实现经典的Faster-RCN, RetinaNet中使用的编码解码方式
    '''
    def __init__(self,weights=[10,10,5,5],scale_clamp=_DEFAULT_SCALE_CLAMP):
        self.weights = weights
        self.scale_clamp = scale_clamp


    def get_deltas(self,boxes,gboxes,labels,indices,img_size=None):
        """
        the labels,indices is the output of matcher
        boxes:[batch_size,N,4]
        gboxes:[batch_size,M,4]
        labels:[batch_size,N]
        indices:[batch_size,N]
        output:
        [batch_size,N,4]
        """
        return tfop.get_boxes_deltas(boxes=boxes,gboxes=gboxes,labels=labels,indices=indices,
                                    scale_weights=self.weights)

    @staticmethod
    def decode_boxes(boxes,deltas,prio_scaling):
        #return tfop.decode_boxes1(boxes=boxes,res=deltas,prio_scaling=prio_scaling)
        return odb.decode_boxes(boxes=boxes,regs=deltas,prio_scaling=prio_scaling)

    def apply_deltas(self,deltas,boxes,img_size=None):
        '''
        :param deltas: [batch_size,N,4]/[N,4]
        :param boxes: [batch_size,N,4]/[N.4]
        :return:
        '''
        B0 = boxes.get_shape().as_list()[0]
        B1 = deltas.get_shape().as_list()[0]
        assert len(deltas.get_shape()) == len(boxes.get_shape()), "deltas and boxes's dims must be equal."

        if len(deltas.get_shape()) == 2:
            return self.decode_boxes(boxes, deltas, prio_scaling=[1/x for x in self.weights])

        if B0==B1:
            return wmlt.static_or_dynamic_map_fn(lambda x:self.decode_boxes(x[0], x[1], prio_scaling=[1/x for x in self.weights]),
                                                 elems=[boxes,deltas],dtype=tf.float32,back_prop=False)
        elif B0==1 and B1>1:
            boxes = tf.squeeze(boxes,axis=0)
            return wmlt.static_or_dynamic_map_fn(lambda x:self.decode_boxes(boxes, x, prio_scaling=[1/x for x in self.weights]),
                                             elems=deltas,dtype=tf.float32,back_prop=False)
        else:
            raise Exception("Error batch size")

class CenterBox2BoxTransform(AbstractBox2BoxTransform):
    '''
    '''
    def __init__(self,num_classes,k,nms_threshold=0.3,gaussian_iou=0.7,dis_threshold=1):
        self.num_classes = num_classes
        self.gaussian_iou = gaussian_iou
        self.k = k
        self.nms_threshold = nms_threshold
        self.dis_threshold = dis_threshold


    def get_deltas(self,gboxes,glabels,glength,output_size):
        """
        gboxes:[batch_size,M,4]
        glabels:[batch_size,M]
        output:
        output_heatmaps_tl: top left heatmaps [B,OH,OW,C]
        output_heatmaps_br: bottom right heatmaps [B,OH,OW,C]
        output_heatmaps_c: center heatmaps [B,OH,OW,C]
        output_offset: positive point offset [B,max_box_nr,6] (ytl,xtl,ybr,xbr,yc,xc)
        output_tags: positive point index [B,max_box_nr,3] (itl,ibr,ic)
        """
        g_heatmaps_tl, g_heatmaps_br, g_heatmaps_c, g_offset, g_tags = tfop.center_boxes_encode(gboxes,
                                                                                           glabels,
                                                                                           glength,
                                                                                           output_size,
                                                                                           self.num_classes,
                                                                                           max_box_nr=-1,
                                                                                           gaussian_iou=self.gaussian_iou)
        outputs = {}
        outputs['g_heatmaps_tl'] = g_heatmaps_tl
        outputs['g_heatmaps_br'] = g_heatmaps_br
        outputs['g_heatmaps_ct'] = g_heatmaps_c
        outputs['g_offset'] = g_offset
        outputs['g_index'] = g_tags
        max_box_nr = tf.reduce_max(glength)
        outputs['g_index_mask']= tf.sequence_mask(glength,maxlen=max_box_nr)

        return outputs


    @staticmethod
    def pixel_nms(heat,kernel=[3,3],threshold=0.3):
        hmax=tf.nn.max_pool(heat,ksize=[1]+kernel+[1],strides=[1,1,1,1],padding='SAME')
        mask=tf.cast(tf.logical_and(tf.equal(hmax,heat),tf.greater(hmax,threshold)),tf.float32)
        return mask*heat

    @staticmethod
    @wmlt.add_name_scope
    def _topk(scores,k=100):
        B,H,W,C = wmlt.combined_static_and_dynamic_shape(scores)
        scores = tf.reshape(scores,[B,-1])
        topk_scores,topk_inds = tf.nn.top_k(scores,k=k)
        topk_classes = topk_inds%C
        topk_inds = topk_inds//C
        topk_ys = tf.cast(topk_inds//W,tf.float32)
        topk_xs = tf.cast(topk_inds%W,tf.float32)
        return topk_scores,topk_inds,topk_classes,topk_ys,topk_xs


    @wmlt.add_name_scope
    def apply_deltas(self,datas,num_dets,img_size=None):
        '''
        '''
        h_tl = tf.nn.sigmoid(datas['heatmaps_tl'])
        h_br  = tf.nn.sigmoid(datas['heatmaps_br'])
        h_ct = tf.nn.sigmoid(datas['heatmaps_ct'])

        B,H,W,C = wmlt.combined_static_and_dynamic_shape(h_tl)

        h_tl = self.pixel_nms(h_tl, threshold=self.nms_threshold)
        h_br = self.pixel_nms(h_br, threshold=self.nms_threshold)
        h_ct = self.pixel_nms(h_ct, threshold=self.nms_threshold)
        tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = self._topk(h_tl, K=self.k)
        br_scores, br_inds, br_clses, br_ys, br_xs = self._topk(h_br, K=self.k)
        ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = self._topk(h_ct, K=self.k)
        K = self.k
        tl_ys = tf.tile(tf.reshape(tl_ys,[B,K,1]),[1,1,K])
        tl_xs = tf.tile(tf.reshape(tl_xs,[B,K,1]),[1,1,K])
        br_ys = tf.tile(tf.reshape(br_ys,[B,1,K]),[1,K,1])
        br_xs = tf.tile(tf.reshape(br_xs,[B,1,K]),[1,K,1])
        ct_ys = tf.reshape(ct_ys,[B,K])
        ct_xs = tf.reshape(ct_xs,[B,K])
        tl_regr = wmlt.batch_gather(datas['offset_tl'],tl_inds)
        br_regr = wmlt.batch_gather(datas['offset_br'],br_inds)
        ct_regr = wmlt.batch_gather(datas['offset_ct'],br_inds)
        tl_regr = tf.reshape(tl_regr,[B,K,1,2])
        br_regr = tf.reshape(br_regr,[B,1,K,2])
        ct_regr = tf.reshape(ct_regr,[B,K,2])
        tl_xs = tl_xs + tl_regr[...,0]
        tl_ys = tl_ys + tl_regr[...,1]
        br_xs = br_xs + br_regr[...,0]
        br_ys = br_ys + br_regr[...,1]
        ct_xs = ct_xs + ct_regr[...,0]
        ct_ys = ct_ys + ct_regr[...,1]

        bboxes = tf.stack([tl_ys,tl_xs,br_ys,br_xs],axis=-1)
        tl_tag = wmlt.batch_gather(datas['tag_tl'],tl_inds)
        br_tag = wmlt.batch_gather(datas['tag_br'],br_inds)
        dists = tf.abs(tl_tag-br_tag)
        dis_inds = (dists>self.dis_threshold)

        tl_scores = tf.tile(tf.reshape(tl_scores,K,1),[1,1,K])
        br_scores = tf.tile(tf.reshape(br_scores,K,1),[1,1,K])
        scores = (tl_scores+br_scores)/2

        tl_clses = tf.tile(tf.reshape(tl_clses,K,1),[1,1,K])
        br_clses = tf.tile(tf.reshape(br_clses,K,1),[1,1,K])
        cls_inds = tf.not_equal(tl_clses,br_clses)

        width_inds = (br_xs<tl_xs)
        height_inds = (br_ys<tl_ys)

        ct = tf.stack([ct_xs, ct_ys], axis=-1)
        center_inds = tfop.center_filter(bboxes,ct,sizes=[],nr=[3,5])

        all_inds = tf.logical_or(cls_inds,dis_inds)
        all_inds = tf.logical_or(all_inds,width_inds)
        all_inds = tf.logical_or(all_inds,height_inds)
        scores = tf.where(all_inds,tf.zeros_like(scores),scores)
        scores,inds = tf.nn.top_k(tf.reshape(scores,[B,-1]),num_dets)

        bboxes = tf.reshape(bboxes,[B,-1,4])
        bboxes = wmlt.batch_gather(bboxes,inds)

        clses = tf.reshape(tl_clses,[B,-1,1])
        clses = wmlt.batch_gather(clses,inds)

        tl_scores = tf.reshape(tl_scores,[B,-1,1])
        tl_scores = wmlt.batch_gather(tl_scores,inds)

        br_scores = tf.reshape(br_scores,[B,-1,1])
        br_scores = wmlt.batch_gather(br_scores,inds)

        return bboxes,scores,tl_scores,br_scores,clses,


class FCOSBox2BoxTransform(AbstractBox2BoxTransform):
    '''
    '''
    def __init__(self,num_classes=None):
        self.num_classes = num_classes


    def get_deltas(self,gboxes,glabels,glength,min_size,max_size,fm_shape,img_size):
        """
        gboxes:[batch_size,M,4]
        glabels:[batch_size,M]
        output:
        output_heatmaps_tl: top left heatmaps [B,OH,OW,C]
        output_heatmaps_br: bottom right heatmaps [B,OH,OW,C]
        output_heatmaps_c: center heatmaps [B,OH,OW,C]
        output_offset: positive point offset [B,max_box_nr,6] (ytl,xtl,ybr,xbr,yc,xc)
        output_tags: positive point index [B,max_box_nr,3] (itl,ibr,ic)
        """
        if self.num_classes is not None:
            glabels = tf.clip_by_value(glabels,0,self.num_classes)
        g_regression,g_center_ness,gt_boxes,gt_classes = tfop.fcos_boxes_encode(gbboxes=gboxes,
                                                                               glabels=tf.cast(glabels,tf.int32),
                                                                               glength=glength,
                                                                               img_size=img_size,
                                                                               fm_shape=fm_shape,
                                                                               min_size=min_size,
                                                                               max_size=max_size)
        outputs = {}
        outputs['g_regression'] = g_regression
        outputs['g_center_ness'] = g_center_ness
        outputs['g_boxes'] = gt_boxes
        outputs['g_classes'] = gt_classes

        return outputs

    @wmlt.add_name_scope
    def apply_deltas(self,regression,img_size=None,fm_size=None):
        if len(regression.get_shape()) == 2:
            B = 1
            H = fm_size[0]
            W = fm_size[1]

        elif len(regression.get_shape()) == 4:
            B,H,W,_ = wmlt.combined_static_and_dynamic_shape(regression)
        else:
            raise NotImplementedError("Error")
        x_i,y_i = tf.meshgrid(tf.range(W),tf.range(H))
        if isinstance(img_size,tf.Tensor) and img_size.dtype != tf.float32:
            img_size = tf.to_float(img_size)
        H = tf.to_float(H)
        W = tf.to_float(W)
        y_f = tf.to_float(y_i)+0.5
        x_f = tf.to_float(x_i)+0.5
        y_delta = img_size[0]/H
        x_delta = img_size[1]/W
        y_base_value = y_f*y_delta
        x_base_value = x_f*x_delta
        base_value = tf.stack([y_base_value,x_base_value,y_base_value,x_base_value],axis=-1)
        if len(regression.get_shape())==4:
            base_value = tf.expand_dims(base_value,axis=0)
            base_value = tf.stop_gradient(tf.tile(base_value,[B,1,1,1]))
            multi = tf.convert_to_tensor([[[[-1,-1,1,1]]]],dtype=tf.float32)
        elif len(regression.get_shape()) == 2:
            base_value = tf.reshape(base_value,[-1,4])
            multi = tf.convert_to_tensor([[-1,-1,1,1]],dtype=tf.float32)


        return base_value+regression*multi

class OffsetBox2BoxTransform(AbstractBox2BoxTransform):
    def __init__(self,scale=True,is_training=True,const_scale=1.0):
        print(f'Test only............................................')
        self.scale = scale
        self.is_training = is_training
        self.const_scale = const_scale
    def get_deltas(self,boxes,gboxes,labels,indices,img_size=None):
        """
        the output is the offset of left-top corner and bottom-right corner
        the labels,indices is the output of matcher
        boxes:[batch_size,N,4]
        gboxes:[batch_size,M,4]
        labels:[batch_size,N]
        indices:[batch_size,N]
        output:
        [batch_size,N,4]
        """
        with tf.name_scope("get_deltas"):
            rgtboxes = wmlt.batch_gather(gboxes,tf.nn.relu(indices))
            deltas = (rgtboxes - boxes)/self.const_scale
            if self.scale:
                scale = tf.sqrt(odb.box_area(boxes))
                deltas = deltas/tf.expand_dims(scale,axis=-1)

            return deltas

    def apply_deltas(self,deltas,boxes,img_size=None):
        '''
        :param deltas: [batch_size,N,4]/[N,4]
        :param boxes: [batch_size,N,4]/[N.4]
        :return:
        '''
        with tf.name_scope("get_deltas"):
            if self.scale:
                scale = tf.sqrt(odb.box_area(boxes))
                deltas = deltas*tf.expand_dims(scale,axis=-1)
            gtboxes = deltas*self.const_scale+boxes
            if not self.is_training:
                gtboxes = tf.clip_by_value(gtboxes,0,1.0)
            ymin,xmin,ymax,xmax = tf.unstack(gtboxes,axis=-1)
            ymax = tf.maximum(ymin,ymax)
            xmax = tf.maximum(xmin,xmax)
            gtboxes = tf.stack([ymin,xmin,ymax,xmax],axis=-1)
            return gtboxes

class CenterNet2Box2BoxTransform(AbstractBox2BoxTransform):
    '''
    '''
    def __init__(self,num_classes,k,score_threshold=0.1,gaussian_iou=0.7,dis_threshold=1):
        self.num_classes = num_classes
        self.gaussian_iou = gaussian_iou
        self.k = k
        self.score_threshold = score_threshold
        self.dis_threshold = dis_threshold
        self.use_custom_op = False


    def get_deltas(self,gboxes,glabels,glength,output_size):
        """
        gboxes:[batch_size,M,4]
        glabels:[batch_size,M]
        output:
        output_heatmaps_tl: top left heatmaps [B,OH,OW,C]
        output_heatmaps_br: bottom right heatmaps [B,OH,OW,C]
        output_heatmaps_c: center heatmaps [B,OH,OW,C]
        output_offset: positive point offset [B,max_box_nr,6] (ytl,xtl,ybr,xbr,yc,xc)
        output_tags: positive point index [B,max_box_nr,3] (itl,ibr,ic)
        """
        g_heatmaps_c, hw_offset, mask = tfop.center2_boxes_encode(gboxes,
                                                                 glabels,
                                                                 glength,
                                                                 output_size,
                                                                 self.num_classes,
                                                                 gaussian_iou=self.gaussian_iou)
        hw,offset = tf.split(hw_offset,2,axis=-1)
        offset_mask,hw_mask = tf.split(mask,2,axis=-1)
        outputs = {}
        outputs['g_heatmaps_ct'] = g_heatmaps_c
        outputs['g_offset'] = offset
        outputs['g_hw'] = hw
        outputs['g_hw_mask'] = hw_mask
        outputs['g_offset_mask'] = offset_mask
        return outputs


    @staticmethod
    def pixel_nms(heat,kernel=[3,3],threshold=0.3):
        hmax=tf.nn.max_pool(heat,ksize=[1]+kernel+[1],strides=[1,1,1,1],padding='SAME')
        mask=tf.cast(tf.logical_and(tf.equal(hmax,heat),tf.greater(hmax,threshold)),tf.float32)
        return mask*heat

    @staticmethod
    @wmlt.add_name_scope
    def _topk(scores,k=100):
        B,H,W,C = wmlt.combined_static_and_dynamic_shape(scores)
        scores = tf.reshape(scores,[B,-1])
        topk_scores,topk_inds = tf.nn.top_k(scores,k=k)
        topk_classes = (topk_inds%C)+1
        topk_inds = topk_inds//C
        topk_ys = tf.cast(topk_inds//W,tf.float32)
        topk_xs = tf.cast(topk_inds%W,tf.float32)
        return topk_scores,topk_inds,topk_classes,topk_ys,topk_xs

    @wmlt.add_name_scope
    def apply_deltas(self,datas,img_size=None):
        if not self.use_custom_op:
            return self.pyapply_deltas(datas,img_size)
        else:
            h_ct = tf.nn.sigmoid(datas['heatmaps_ct'])
            offset = datas['offset']
            hw = datas['hw']
            bboxes, labels, probs, index, lens = tfop.center2_boxes_decode(heatmaps=h_ct,
                                                                          offset=offset,
                                                                          hw=hw,
                                                                          k=self.k,
                                                                          threshold=self.score_threshold)
            return bboxes,labels,probs,index

    @wmlt.add_name_scope
    def pyapply_deltas(self,datas,img_size=None):
        '''
        '''
        h_ct = tf.nn.sigmoid(datas['heatmaps_ct'])
        offset = datas['offset']
        hw = datas['hw']

        B,H,W,C = wmlt.combined_static_and_dynamic_shape(h_ct)
        offset = tf.reshape(offset,[B,-1,2])
        hw = tf.reshape(hw,[B,-1,2])

        h_ct = self.pixel_nms(h_ct, threshold=self.score_threshold)
        ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = self._topk(h_ct, k=self.k)
        C = btf.channel(h_ct)
        hw_inds = ct_inds//C
        K = self.k
        ct_ys = tf.reshape(ct_ys,[B,K])
        ct_xs = tf.reshape(ct_xs,[B,K])
        offset = wmlt.batch_gather(offset,hw_inds)
        offset = tf.reshape(offset,[B,K,2])
        offset_y,offset_x = tf.unstack(offset,axis=-1)
        ct_xs = ct_xs+offset_x
        ct_ys = ct_ys+offset_y
        hw = wmlt.batch_gather(hw,hw_inds)
        hw = tf.reshape(hw,[B,K,2])
        h,w = tf.unstack(hw,axis=-1)
        ymin,xmin,ymax,xmax = [ct_ys-h/2,ct_xs-w/2,ct_ys+h/2,ct_xs+w/2]
        bboxes = tf.stack([ymin,xmin,ymax,xmax],axis=-1)
        bboxes = odb.tfabsolutely_boxes_to_relative_boxes(bboxes,width=W,height=H)

        return bboxes,ct_clses,ct_scores,hw_inds
