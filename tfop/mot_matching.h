#pragma once
#include "strack.h"
#include <vector>
#include <utility>
#include "kalman_filter.h"

namespace MOT
{
    Eigen::MatrixXf embedding_distance(const STrackPtrs_t& tracks, const STrackPtrs_t& detections);
    Eigen::MatrixXf iou_distance(const STrackPtrs_t& tracks, const STrackPtrs_t& detections);
    void fuse_motion(KalmanFilterPtr_t kf,const STrackPtrs_t& tracks, const STrackPtrs_t& detections,Eigen::MatrixXf& cost_matrix,bool only_position=false,float lambda=0.98);
    float cosine_dis(const Eigen::VectorXf& va, const Eigen::VectorXf& vb);
    /*
     * ymin,xmin,ymax,xmax format
     */
    float iou_dis(const BBox_t& va, const BBox_t& vb);

    void linear_assignment(const Eigen::MatrixXf& cost_matrix,float thresh,std::vector<std::pair<int,int>>* matches,std::vector<int>* unmatched_a,std::vector<int>* unmatched_b);
    void linear_assignment(float* data,int data_nr,float* thresh,std::vector<std::pair<int,int>>* matches,std::vector<int>* unmatched_a=nullptr,std::vector<int>* unmatched_b=nullptr);
}
