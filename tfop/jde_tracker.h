#pragma once
#include <Eigen/Core>
#include "strack.h"
#include <vector>

namespace MOT
{
    class JDETracker
    {
        public:
            /*
             * assignment_thresh: if use embds:[iou dis threshold for tracked and detections, iou dis threshold for untracked and detections, embds dis for tracked and detections]
             * if not use embds,[iou dis threshold for tracked and detections, iou dis threshold for untracked and detections]
             */
            JDETracker(float det_thresh=0.1,int frame_rate=30,int track_buffer=30,const std::vector<float>& assignment_thresh={0.5f,0.7f,0.4f});
            STrackPtrs_t update(const BBoxes_t& bboxes,const Probs_t& probs, const Embeddings_t& embds,bool return_losted=false);
            STrackPtrs_t update(const BBoxes_t& bboxes,const Probs_t& probs);
        private:
            STrackPtrs_t joint_stracks(STrackPtrs_t& va,STrackPtrs_t& vb);
            //STrackPtrs_t sub_stracks(STrackPtrs_t& va,STrackPtrs_t& vb);
            //STrackPtrs_t remove_duplicate_stracks(STrackPtrs_t& va,STrackPtrs_t& vb);
        private:
           float              det_thresh_        = 0.1;
           int                frame_rate_        = 30;
           int                track_buffer_      = 30;
           int                frame_id_          = 0;
           int                buffer_size_       = 30;
           int                max_time_lost_     = 30;
           std::vector<float> assignment_thresh_;
           STrackPtrs_t       tracked_stracks_;
           STrackPtrs_t       lost_stracks_;
           STrackPtrs_t       removed_stracks_;
           KalmanFilterPtr_t  kalman_filter_;
    };
}
