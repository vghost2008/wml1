# coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
import wtfop.wtfop_ops as wop
from object_detection2.standard_names import *
import wmodule
from object_detection2.datadef import *
from object_detection2.config.config import global_cfg
from object_detection2.modeling.build import HEAD_OUTPUTS
import object_detection2.odtools as odtl
import object_detection2.keypoints as kp
import wsummary
import basic_tftools as btf

@HEAD_OUTPUTS.register()
class OpenPoseOutputs(wmodule.WChildModule):
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

    @btf.add_name_scope
    def _get_ground_truth(self):
        map_shape = btf.combined_static_and_dynamic_shape(self.pred_maps[0][0])
        output = wop.open_pose_encode(keypoints=self.gt_keypoints,
                                      output_size=map_shape[1:3],
                                      glength=self.gt_length,
                                      keypoints_pair=self.cfg.POINTS_PAIRS,
                                      l_delta=self.cfg.OPENPOSE_L_DELTA,
                                      gaussian_delta=self.cfg.OPENPOSE_GAUSSIAN_DELTA)
        gt_conf_maps = output[0]
        gt_paf_maps = output[1]
        wsummary.feature_map_summary(gt_conf_maps,"gt_conf_maps",max_outputs=5)
        wsummary.feature_map_summary(gt_paf_maps,"gt_paf_maps",max_outputs=5)
        if self.cfg.USE_LOSS_MASK:
            B,H,W,_ = btf.combined_static_and_dynamic_shape(gt_paf_maps)
            image = tf.zeros([B,H,W,1])
            mask = odtl.batch_fill_bboxes(image,self.gt_boxes,v=1.0,
                                          length=self.gt_length,
                                          H=H,
                                          W=W,
                                          relative_coord=True)
            conf_mask = mask
            paf_mask = mask
            tf.summary.image("bboxes_mask",mask,max_outputs=5)
        else:
            conf_mask = None
            paf_mask = None
        return gt_paf_maps, gt_conf_maps,paf_mask,conf_mask

    @wmlt.add_name_scope
    def losses(self):
        """
        Args:

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only.
        """
        assert len(self.pred_maps[0][0].get_shape()) == 4, "error logits dim"
        pred_paf_maps,pred_conf_maps,pred_finaly_maps = self.pred_maps
        gt_paf_maps, gt_conf_maps,paf_mask,conf_mask = self._get_ground_truth()
        loss_pafs = []
        loss_confs = []
        fgt_nr = tf.cast(self.gt_length,tf.float32)+0.1
        for i,map in enumerate(pred_paf_maps):
            loss = tf.losses.mean_squared_error(labels=gt_paf_maps,
                                                    predictions=map,loss_collection=None,
                                                    reduction=tf.losses.Reduction.NONE)
            if paf_mask is not None:
                loss = loss*paf_mask
            loss = tf.reduce_sum(loss,axis=[1,2,3])/fgt_nr
            loss = tf.reduce_sum(loss)
            tf.summary.scalar(f"openpose_paf_loss{i}",loss)
            loss_pafs.append(loss)

        for i,map in enumerate(pred_conf_maps):
            loss = tf.losses.mean_squared_error(labels=gt_conf_maps,
                                                predictions=map,loss_collection=None,
                                                reduction=tf.losses.Reduction.NONE)
            if conf_mask is not None:
                loss = loss*conf_mask
            loss = tf.reduce_sum(loss,axis=[1,2,3])/fgt_nr
            loss = tf.reduce_sum(loss)
            tf.summary.scalar(f"openpose_conf_loss{i}",loss)
            loss_confs.append(loss)

        loss_pafs = tf.add_n(loss_pafs)
        loss_confs = tf.add_n(loss_confs)
        return {"loss_pafs": loss_pafs, "loss_confs": loss_confs}

    @wmlt.add_name_scope
    def inference(self, inputs, pred_maps):
        """
        Arguments:
            inputs: same as forward's batched_inputs
            pred_maps: outputs of openpose head
        Returns:
            results:
            RD_BOXES: [B,N,4]
            RD_PROBABILITY:[ B,N]
            RD_KEYPOINTS:[B,N,NUM_KEYPOINTS,2]
            RD_LENGTH:[B]
        """
        _,_,pred_finaly_maps = pred_maps
        C = btf.channel(pred_finaly_maps)
        conf_maps,paf_maps = tf.split(pred_finaly_maps,
                                      [self.cfg.NUM_KEYPOINTS,C-self.cfg.NUM_KEYPOINTS],
                                      axis=-1)
        output_keypoints,output_lens = wop.open_pose_decode(conf_maps,paf_maps,self.cfg.POINTS_PAIRS,
                                                keypoints_th=self.cfg.OPENPOSE_KEYPOINTS_TH,
                                                interp_samples=self.cfg.OPENPOSE_INTERP_SAMPLES,
                                                paf_score_th=self.cfg.OPENPOSE_PAF_SCORE_TH,
                                                conf_th=self.cfg.DET_SCORE_THRESHOLD_TEST,
                                                max_detection=self.max_detections_per_image
                                                )

        bboxes = kp.batch_get_bboxes(output_keypoints,output_lens)
        outdata = {RD_BOXES: bboxes, RD_LENGTH: output_lens,
                   RD_KEYPOINT: output_keypoints}
        if global_cfg.GLOBAL.SUMMARY_LEVEL <= SummaryLevel.DEBUG:
            wsummary.keypoints_image_summary(images=inputs[IMAGE],
                                             keypoints=output_keypoints,
                                             lengths=outdata[RD_LENGTH],
                                             keypoints_pair=self.cfg.POINTS_PAIRS,
                                             name="KeyPoints_result")
        return outdata

