# coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
import tfop
from object_detection2.standard_names import *
import wmodule
from object_detection2.datadef import *
from object_detection2.config.config import global_cfg
from object_detection2.modeling.build import HEAD_OUTPUTS
import object_detection2.odtools as odtl
import object_detection2.keypoints as kp
import wsummary
import basic_tftools as btf
import wnnlayer as wnnl

@HEAD_OUTPUTS.register()
class HRNetPEOutputs(wmodule.WChildModule):
    def __init__(
            self,
            cfg,
            parent,
            pred_maps,
            gt_boxes=None,
            gt_labels=None,
            gt_length=None,
            gt_keypoints=None,
            max_detections_per_image=100,
            **kwargs,
    ):
        """
        Args:
            cfg: Only the child part
            gt_boxes: [B,N,4] (ymin,xmin,ymax,xmax)
            gt_labels: [B,N]
            gt_length: [B]
            gt_keypoints: [B,N,NUM_KEYPOINTS,2] (x,y)
        """
        super().__init__(cfg, parent=parent, **kwargs)
        self.max_detections_per_image = max_detections_per_image
        self.pred_maps = pred_maps
        self.gt_boxes = gt_boxes
        self.gt_labels = gt_labels
        self.gt_length = gt_length
        self.gt_keypoints = gt_keypoints
        self.num_keypoints = cfg.NUM_KEYPOINTS

    @btf.add_name_scope
    def _get_ground_truth(self,net):
        map_shape = btf.combined_static_and_dynamic_shape(net)
        output = tfop.hr_net_encode(keypoints=self.gt_keypoints,
                                   output_size=map_shape[1:3],
                                   glength=self.gt_length,
                                   gaussian_delta=self.cfg.OPENPOSE_GAUSSIAN_DELTA)
        gt_conf_maps = output[0]
        gt_indexs = output[1]
        wsummary.feature_map_summary(gt_conf_maps,"gt_conf_maps",max_outputs=5)
        if self.cfg.USE_LOSS_MASK:
            B,H,W,_ = btf.combined_static_and_dynamic_shape(gt_conf_maps)
            image = tf.zeros([B,H,W,1])
            mask = odtl.batch_fill_bboxes(image,self.gt_boxes,v=1.0,
                                          length=self.gt_length,
                                          H=H,
                                          W=W,
                                          relative_coord=True)
            conf_mask = mask
            tf.summary.image("loss_mask",mask,max_outputs=5)
        else:
            conf_mask = None
        return gt_conf_maps,gt_indexs,conf_mask

    @wmlt.add_name_scope
    def losses(self):
        """
        Args:

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only.
        """
        assert len(self.pred_maps[0].get_shape()) == 4, "error logits dim"
        loss_confs = []
        pull_loss = None
        push_loss = None
        for i,net in enumerate(self.pred_maps):
            gt_conf_maps,indexs,conf_mask = self._get_ground_truth(net)
            if i==0:
                net,tags = tf.split(net,2,axis=-1)
            loss = tf.losses.mean_squared_error(labels=gt_conf_maps,
                                                predictions=net,
                                                loss_collection=None,
                                                reduction=tf.losses.Reduction.NONE)
            if conf_mask is not None:
                loss = loss*conf_mask
            loss = tf.reduce_mean(loss,axis=[1,2,3])
            loss = tf.reduce_sum(loss)
            tf.summary.scalar(f"hr_net_conf{i}",loss)
            loss_confs.append(loss)
            if i==0:
                pull_loss,push_loss = self.ae_loss(tags,indexs)

        loss_confs = tf.add_n(loss_confs)
        ae_loss_scale = self.cfg.AE_LOSS_SCALE
        return {"loss_pull": pull_loss*ae_loss_scale, "loss_push":push_loss*ae_loss_scale,"loss_confs": loss_confs}

    def ae_loss(self,net,indexs):
        pull_loss,push_loss = tf.map_fn(lambda x:self.ae_loss_for_single_img(x[0],x[1],x[2]),
                                        elems=[net,indexs,self.gt_length],
                                        back_prop=True,dtype=(tf.float32,tf.float32))
        pull_loss = tf.reduce_sum(pull_loss)
        push_loss = tf.reduce_sum(push_loss)
        return pull_loss,push_loss

    def ae_loss_for_single_img(self,net,indexs,nr):
        indexs = indexs[:nr]
        fnr = tf.cast(nr,tf.float32)
        with tf.name_scope("pull_loss"):
            net = tf.reshape(net,[-1])
            old_shape = btf.combined_static_and_dynamic_shape(indexs)
            ids = tf.gather(net,tf.nn.relu(tf.reshape(indexs,[-1])))
            ids = tf.reshape(ids,old_shape)
            weights = tf.cast(tf.greater_equal(indexs,0),tf.float32)
            mean = tf.reduce_sum(ids*weights,axis=-1,keepdims=True)/tf.maximum(tf.reduce_sum(weights,axis=-1,keepdims=True),1e-8)
            pull_loss = tf.square(ids-mean)*weights
            pull_loss = tf.reduce_sum(pull_loss)/fnr

        with tf.name_scope("push_loss"):
            range = tf.range(nr)
            X,Y = tf.meshgrid(range,range)
            X = tf.reshape(X,[-1])
            Y = tf.reshape(Y,[-1])
            mask = tf.not_equal(X,Y)
            ftotal_nr = tf.reduce_sum(tf.cast(mask,tf.float32))+1e-8
            X = tf.boolean_mask(X,mask)
            Y = tf.boolean_mask(Y,mask)
            loss = -tf.square(tf.gather(mean,X)-tf.gather(mean,Y))
            loss = tf.exp(loss)
            loss = tf.reduce_sum(loss)
            loss = loss/ftotal_nr
            push_loss = loss*0.5

        return pull_loss,push_loss

    @wmlt.add_name_scope
    def inference(self, inputs, pred_maps):
        """
        Arguments:
            inputs: same as forward's batched_inputs
            pred_maps: output of hrnet head
        Returns:
            results:
            RD_BOXES: [B,N,4]
            RD_PROBABILITY:[ B,N]
            RD_KEYPOINTS:[B,N,NUM_KEYPOINTS,2]
            RD_LENGTH:[B]
        """
        with tf.name_scope("aggregate_results"):
            pred0,det1 = pred_maps
            det0,tags = tf.split(pred0,num_or_size_splits=2,axis=-1)
            target_size = wmlt.combined_static_and_dynamic_shape(det1)[1:3]
            tags = tf.image.resize_bilinear(tags,target_size)
            det0 = tf.image.resize_bilinear(det0,target_size)
            H,W = target_size
            wsummary.feature_map_summary(tags,"tags",max_outputs=5)

            tags = tf.expand_dims(tags,axis=-1)
            det = (det0+det1)/2
            wsummary.feature_map_summary(det0,"det0",max_outputs=5)
            wsummary.feature_map_summary(det1,"det1",max_outputs=5)

        tag_k,loc_k,val_k = self.top_k(det,tags)
        ans = self.match(tag_k,loc_k,val_k)
        ans = self.adjust(ans,det=det)
        ans = tfop.hr_net_refine(ans,det=det,tag=tags)

        scores = ans[...,2]
        scores = tf.reduce_mean(scores,axis=-1,keepdims=False)
        x,y = tf.unstack(ans[...,:2],axis=-1)
        mask = tf.greater(scores,self.cfg.DET_SCORE_THRESHOLD_TEST)
        size = wmlt.combined_static_and_dynamic_shape(x)[1]
        x,output_lens = wmlt.batch_boolean_mask(x,mask,size=size,return_length=True)
        y = wmlt.batch_boolean_mask(y,mask,size=size)
        scores = wmlt.batch_boolean_mask(scores,mask,size=size)
        keypoints = tf.stack([x,y],axis=-1)

        output_keypoints = kp.keypoints_absolute2relative(keypoints,width=W,height=H)
        bboxes = kp.batch_get_bboxes(output_keypoints,output_lens)

        outdata = {RD_BOXES: bboxes, RD_LENGTH: output_lens,
                   RD_KEYPOINT: output_keypoints,
                   RD_PROBABILITY:scores,
                   RD_LABELS:tf.ones_like(scores,dtype=tf.int32)}

        if global_cfg.GLOBAL.SUMMARY_LEVEL <= SummaryLevel.DEBUG:
            wsummary.keypoints_image_summary(images=inputs[IMAGE],
                                             keypoints=output_keypoints,
                                             lengths=outdata[RD_LENGTH],
                                             keypoints_pair=self.cfg.POINTS_PAIRS,
                                             name="keypoints_results")
        return outdata


    @btf.add_name_scope
    def adjust(self,ans,det):
        locs = ans[...,:2]
        values = ans[...,2]
        x,y = tf.unstack(locs,axis=-1)
        org_x,org_y = x,y
        xx = tf.cast(x,tf.int32)
        yy = tf.cast(y,tf.int32)
        B,H,W,num_keypoints = btf.combined_static_and_dynamic_shape(det)
        det = tf.transpose(det,[0,3,1,2])
        det = tf.reshape(det,[B*num_keypoints,H*W])
        yy_p = tf.minimum(yy+1,H-1)
        yy_n = tf.maximum(yy-1,0)
        xx_p = tf.minimum(xx+1,W-1)
        xx_n = tf.maximum(xx-1,0)

        def get_values(_xx,_yy):
            B,N,KN = btf.combined_static_and_dynamic_shape(_xx)
            _xx = tf.transpose(_xx,[0,2,1])
            _yy = tf.transpose(_yy,[0,2,1])
            _xx = tf.reshape(_xx,[B*KN,N])
            _yy = tf.reshape(_yy,[B*KN,N])
            index = _xx+_yy*W
            vals = tf.batch_gather(det,index)
            vals = tf.reshape(vals,[B,KN,N])
            vals = tf.transpose(vals,[0,2,1])
            return vals

        y_p = y+0.25
        y_n = y-0.25
        x_p = x+0.25
        x_n = x-0.25

        y = tf.where(get_values(xx,yy_p)>get_values(xx,yy_n),y_p,y_n)
        x = tf.where(get_values(xx_p,yy)>get_values(xx_n,yy),x_p,x_n)

        x = x+0.5
        y = y+0.5

        x = tf.where(values>0,x,org_x)
        y = tf.where(values>0,y,org_y)

        loc = tf.stack([x,y],axis=-1)

        B,N,KP,C = btf.combined_static_and_dynamic_shape(ans)
        _,data = tf.split(ans,[2,C-2],axis=-1)

        return tf.concat([loc,data],axis=-1)

    @btf.add_name_scope
    def top_k(self,det,tags):
        det = wnnl.pixel_nms(det,kernel=self.cfg.HRNET_PE_NMS_KERNEL)

        max_num_people = self.cfg.HRNET_PE_MAX_NUM_PEOPLE
        B,H,W,num_keypoints = btf.combined_static_and_dynamic_shape(det)
        det = tf.reshape(det,[B,H*W,num_keypoints])
        det = tf.transpose(det,[0,2,1])

        val_k,indices = tf.nn.top_k(det,k=max_num_people)
        B,H,W,num_keypoints,C = btf.combined_static_and_dynamic_shape(tags)
        tags = tf.reshape(tags,[B,H*W,num_keypoints,C])
        tags = tf.transpose(tags,[0,2,1,3])
        tag_k = tf.batch_gather(tags,indices)

        x = indices%W
        y = indices//H

        loc_k = tf.stack([x,y],axis=-1)
        loc_k = tf.cast(loc_k,tf.float32)

        return tag_k,loc_k,val_k

    def match(self,tag_k,loc_k,val_k):
        res = tf.map_fn(lambda x:tfop.match_by_tag(x[0],x[1],x[2],
                                         detection_threshold=self.cfg.HRNET_DETECTION_THRESHOLD,
                                         tag_threshold=self.cfg.HRNET_TAG_THRESHOLD,
                                         use_detection_val=self.cfg.HRNET_USE_DETECTION_VAL),
                        elems=(tag_k,loc_k,val_k),dtype=tf.float32,back_prop=False)
        return res



