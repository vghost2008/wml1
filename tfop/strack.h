#pragma once
#include <Eigen/Core>
#include <vector>
#include <memory>
#include "kalman_filter.h"

namespace MOT
{
    using BBox_t = Eigen::Matrix<float,4,1>;
    using BBoxes_t = Eigen::Matrix<float,Eigen::Dynamic,4,Eigen::RowMajor>;
    using Probs_t = Eigen::Matrix<float,Eigen::Dynamic,1>;
    using Embeddings_t = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

class KalmanFilter;
class STrack;
using STrackPtr_t = std::shared_ptr<STrack>;
using STrackPtrs_t = std::vector<std::shared_ptr<STrack>>;
class STrack
{
    public:
    enum STrackState
    {
        NEW,
        TRACKED,
        LOST,
        REMOVED,
    };
    public:
        /*
         * bboxes: [ymin,xmin,ymax,xmax]
         */
        STrack(const Eigen::VectorXf& bboxes,float score,const Eigen::VectorXf& temp_feat=Eigen::VectorXf(),int buffer_size=30);
        void update_features(const Eigen::VectorXf& feat);
        void predict();
        static void multi_predict(STrackPtrs_t& stracks);
        void activate(std::shared_ptr<KalmanFilter>& kalman_filter,int frame_id);
        void re_activate(const STrack& new_track,int frame_id,bool new_id=false);
        void update(const STrack& new_track,int frame_id,bool update_feature=true);
   public:
        inline auto& mean() { return mean_; }
        inline const auto& mean()const { return mean_; }
        inline auto& covariance() { return covariance_; }
        inline const auto& covariance()const { return covariance_; }
        inline void set_mean(const auto& v) { mean_ = v; }
        inline void set_covariance(const auto& v) { covariance_ = v; }
        BBox_t get_latest_bbox()const;
        BBox_t get_latest_yminxminymaxxmax_bbox()const;
        inline bool is_activated()const { return is_activated_;}
        inline const Eigen::VectorXf& curr_feat()const { return curr_feat_; }
    public:
        inline STrackState state()const { return state_; }
        inline void set_state(STrackState s) { state_ = s; }
        inline void mark_lost() { state_ = LOST; }
        inline void mark_removed() { state_ = REMOVED; }
    public:
        inline float score()const { return score_; }
        inline int end_frame()const { return frame_id_; }
        inline int track_id()const { return track_id_; }
    private:
        static int g_id_count_;
        static inline int next_id() { return g_id_count_++; }
        static inline void reset_id() { g_id_count_ = 0; }
    private:
        BBox_t bbox_;  //cx,cy,a,h
        Mean_t mean_;  //cx,cy,a,h,vcx,vcy,...
        Cov_t covariance_;
        Eigen::VectorXf smooth_feat_;
        Eigen::VectorXf curr_feat_;
        std::vector<Eigen::VectorXf> features_;
        int tracklet_len_ = 0;
        int frame_id_ = 0;
        int start_frame_ = 0;
        int buffer_size_ = 0;
        int track_id_;
        STrackState state_;
        std::shared_ptr<KalmanFilter> kalman_filter_;
        float score_ = 0.0f;
        const float alpha_ = 0.9;
        bool is_activated_ = false;

};
}
