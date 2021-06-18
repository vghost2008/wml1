#include "strack.h"
#include "kalman_filter.h"
#include "track_toolkit.h"
#include <iostream>

using namespace std;
using namespace MOT;
int STrack::g_id_count_ = 0;
STrack::STrack(const Eigen::VectorXf& bboxes,float score,const Eigen::VectorXf& temp_feat,int buffer_size)
:score_(score)
,buffer_size_(buffer_size)
{
    const auto w = (bboxes[3]-bboxes[1]);
    const auto h = (bboxes[2]-bboxes[0]);

    bbox_[0] = bboxes[1]+w/2;
    bbox_[1] = bboxes[0]+h/2;
    bbox_[2] = w/max<float>(h,1e-4);
    bbox_[3] = h;

    update_features(temp_feat);

    mean_.setZero();
    mean_.block<4,1>(0,0) = bbox_;
}
void STrack::update_features(const Eigen::VectorXf& feat)
{
    if(feat.size()==0) 
        return;

    curr_feat_ = feat;

    if(smooth_feat_.size() == 0)
        smooth_feat_ = feat;
    else
        smooth_feat_ = alpha_*smooth_feat_+(1.0f-alpha_)*feat;
    smooth_feat_ = l2_normalize(smooth_feat_);
}
void STrack::predict()
{
    auto mean = mean_;
    if(state_ != STrackState::TRACKED)
        mean[7] = 0;
    tie(mean_,covariance_) = kalman_filter_->predict(mean,covariance_);
}
void STrack::multi_predict(STrackPtrs_t& stracks)
{
    for(auto& track:stracks) {
        track->predict();
    }
}
void STrack::activate(shared_ptr<KalmanFilter>& kalman_filter,int frame_id)
{
    if(frame_id<0) {
        cout<<"ERROR frame id "<<frame_id<<", min frame id allowed is 1."<<endl;
        frame_id = 1;
    }
    kalman_filter_ = kalman_filter;
    track_id_ = next_id();

    tie(mean_,covariance_) = kalman_filter_->initiate(bbox_);

    tracklet_len_ = 0;
    state_ = STrackState::TRACKED;

    if(1 == frame_id)
        is_activated_ = true;
    frame_id_ = frame_id;
    start_frame_ = frame_id;
}
void STrack::re_activate(const STrack& new_track,int frame_id,bool new_id)
{
    tie(mean_,covariance_) = kalman_filter_->update(mean_,covariance_,get_latest_bbox());

    update_features(new_track.curr_feat_);

    tracklet_len_ = 0;
    state_ = STrackState::TRACKED;
    is_activated_ = true;
    frame_id_ = frame_id;
    if(new_id)
        track_id_ = next_id();
}
void STrack::update(const STrack& new_track,int frame_id,bool update_feature)
{
    frame_id_ = frame_id;
    tracklet_len_ += 1;

    tie(mean_,covariance_) = kalman_filter_->update(mean_,covariance_,new_track.get_latest_bbox());

    state_ = STrack::TRACKED;
    is_activated_ = true;
    score_ = new_track.score_;
    if(update_feature)
        update_features(new_track.curr_feat_);
}
BBox_t STrack::get_latest_bbox()const
{
    return mean_.block<4,1>(0,0);
}
BBox_t STrack::get_latest_yminxminymaxxmax_bbox()const 
{
    auto bbox = get_latest_bbox();
    BBox_t res;
    auto hw = bbox[2]*bbox[3]/2;
    auto hh = bbox[3]/2;

    res[0] = bbox[1]-hh;
    res[1] = bbox[0]-hw;
    res[2] = bbox[1]+hh;
    res[3] = bbox[0]+hw;

    return res;
}
