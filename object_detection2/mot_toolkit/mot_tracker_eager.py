import tfop
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

class FairMOTTracker:
    def __init__(self,det_thredh=0.1,frame_rate=25,track_buffer=10,assignment_thresh=[0.9,0.9,0.2],return_losted=False):
        self.det_thredh = det_thredh
        self.frame_rate = frame_rate
        self.track_buffer = track_buffer
        self.assignment_thresh = assignment_thresh
        self.return_losted = return_losted
        self.lost_stracks = []

    def __call__(self,bboxes,probs,embds,is_first_frame):
        tracked_id, tracked_bboxes, tracked_idx = tfop.fair_mot(bboxes, probs,embds,
                                                                    is_first_frame=is_first_frame,
                                                                    det_thredh=self.det_thredh,
                                                                    frame_rate=self.frame_rate,
                                                                    track_buffer=self.track_buffer,
                                                                    assignment_thresh=self.assignment_thresh,
                                                                    return_losted=self.return_losted)
        return tracked_id.numpy(), tracked_bboxes.numpy(), tracked_idx.numpy()

class SORTMOTTracker:
    def __init__(self,det_thredh=0.1,frame_rate=25,track_buffer=10,assignment_thresh=[0.9,0.9,0.2],return_losted=False):
        self.det_thredh = det_thredh
        self.frame_rate = frame_rate
        self.track_buffer = track_buffer
        self.assignment_thresh = assignment_thresh
        self.return_losted = return_losted
        self.lost_stracks = []

    def __call__(self,bboxes,probs=None,is_first_frame=False):
        if probs is None:
            nr = bboxes.shape[0]
            probs = np.ones([nr],dtype=np.float32)
        tracked_id, tracked_bboxes, tracked_idx = tfop.sort_mot(bboxes, probs,is_first_frame,
                                                                    det_thredh=self.det_thredh,
                                                                    frame_rate=self.frame_rate,
                                                                    track_buffer=self.track_buffer,
                                                                    assignment_thresh=self.assignment_thresh)
        tracked_bboxes = tracked_bboxes.numpy()
        tracked_bboxes = np.maximum(tracked_bboxes,0)
        return tracked_id.numpy(), tracked_bboxes, tracked_idx.numpy()

