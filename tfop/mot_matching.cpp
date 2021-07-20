#include "mot_matching.h"
#include "bboxes.h"
#include "lap.h"

using namespace std;
using namespace Eigen;

namespace MOT {
Eigen::MatrixXf embedding_distance(const STrackPtrs_t& tracks, const STrackPtrs_t& detections)
{
    MatrixXf cost_matrix(tracks.size(),detections.size());

    for(auto i=0; i<tracks.size(); ++i) {
        for(auto j=0; j<detections.size(); ++j) {
            auto tf = tracks[i]->curr_feat();
            auto df = detections[j]->curr_feat();
            cost_matrix(i,j) = cosine_dis(tf,df);
        }
    }

    return cost_matrix;
}
Eigen::MatrixXf iou_distance(const STrackPtrs_t& tracks, const STrackPtrs_t& detections)
{
    MatrixXf cost_matrix(tracks.size(),detections.size());

    for(auto i=0; i<tracks.size(); ++i) {
        for(auto j=0; j<detections.size(); ++j) {
            auto tbbox = tracks[i]->get_latest_yminxminymaxxmax_bbox();
            auto dbbox = detections[j]->get_latest_yminxminymaxxmax_bbox();
            cost_matrix(i,j) = iou_dis(tbbox,dbbox);
        }
    }

    return cost_matrix;
}
float cosine_dis(const Eigen::VectorXf& va, const Eigen::VectorXf& vb)
{
    auto v0 = va.dot(vb);
    return 1.0-v0/max<float>(1e-8,va.norm()*vb.norm());
}
float iou_dis(const BBox_t& va, const BBox_t& vb)
{
    return 1.0-bboxes_jaccardv1(va,vb);
}
void linear_assignment(const Eigen::MatrixXf& cost_matrix,float thresh,vector<pair<int,int>>* matches,vector<int>* unmatched_a,vector<int>* unmatched_b)
{
    matches->clear();
    unmatched_a->clear();
    unmatched_b->clear();

    if(cost_matrix.size() == 0) {
        for(auto i=0; i<cost_matrix.rows(); ++i) 
            unmatched_a->push_back(i);
        for(auto i=0; i<cost_matrix.cols(); ++i) 
            unmatched_b->push_back(i);
        return;
    }

    float max_cost = cost_matrix.sum()+1.0f;
    auto data_nr = max(cost_matrix.cols(),cost_matrix.rows());
    Matrix<float,Dynamic,Dynamic,RowMajor> new_cost_matrix(data_nr,data_nr);

    new_cost_matrix.setConstant(max_cost);

    for(auto i=0; i<cost_matrix.cols(); ++i) {
        for(auto j=0; j<cost_matrix.rows(); ++j) {
            if(cost_matrix(j,i)<= thresh) {
                new_cost_matrix(j,i) = cost_matrix(j,i);
            }
        }
    }

    auto assign_cost= unique_ptr<int[]>(new int[data_nr]);
    auto rowsol = unique_ptr<int[]>(new int[data_nr]);
    auto colsol = unique_ptr<int[]>(new int[data_nr]);
    auto u = unique_ptr<float[]>(new float[data_nr]);
    auto v = unique_ptr<float[]>(new float[data_nr]);

    lap<false>(data_nr,new_cost_matrix.data(),false,rowsol.get(),colsol.get(),u.get(),v.get());

    for(auto i=0; i<data_nr; ++i) {
        auto j = rowsol.get()[i];
        if(new_cost_matrix(i,j)<thresh) {
            matches->emplace_back(make_pair(i,j));
        } else {
            if(i<cost_matrix.rows())
                unmatched_a->push_back(i);
            if(j<cost_matrix.cols())
                unmatched_b->push_back(j);
        }
    }
}
void linear_assignment(float* data,int data_nr,float* thresh,std::vector<std::pair<int,int>>* matches,std::vector<int>* unmatched_a,std::vector<int>* unmatched_b)
{
    matches->clear();
    if(nullptr != unmatched_a)
        unmatched_a->clear();
    if(nullptr != unmatched_b)
        unmatched_b->clear();

    auto assign_cost= unique_ptr<int[]>(new int[data_nr]);
    auto rowsol = unique_ptr<int[]>(new int[data_nr]);
    auto colsol = unique_ptr<int[]>(new int[data_nr]);
    auto u = unique_ptr<float[]>(new float[data_nr]);
    auto v = unique_ptr<float[]>(new float[data_nr]);
    auto new_cost_matrix = (float(*)[data_nr])(data);

    lap<false>(data_nr,data,false,rowsol.get(),colsol.get(),u.get(),v.get());

    if(nullptr != thresh) {
        for(auto i=0; i<data_nr; ++i) {
            auto j = rowsol.get()[i];
            if(new_cost_matrix[i][j]<*thresh) {
                matches->emplace_back(make_pair(i,j));
            } else {
                unmatched_a->push_back(i);
                unmatched_b->push_back(j);
            }
        }
    } else {
        for(auto i=0; i<data_nr; ++i) {
            auto j = rowsol.get()[i];
            matches->emplace_back(make_pair(i,j));
        }
    }
}
void fuse_motion(KalmanFilterPtr_t kf,const STrackPtrs_t& tracks, const STrackPtrs_t& detections,Eigen::MatrixXf& cost_matrix,bool only_position,float lambda)
{
    if(cost_matrix.size() == 0)
        return;

    auto       gating_dim              = only_position?2:4;
    auto       gating_threshold        = chi2inv95[gating_dim];
    const auto kMaxCost                = 1.0f;
    MatrixXf   measurements(detections.size(),4);

    for(auto i=0; i<detections.size(); ++i) {
        measurements.block<1,4>(i,0) = detections[i]->get_latest_bbox().transpose();
    }

    for(auto i=0; i<tracks.size(); ++i) {
        auto& track = tracks[i];
        auto gating_distance = kf->gating_distance(track->mean(),track->covariance(),measurements,only_position,KalmanFilter::MAHA);

        for(auto j=0; j<detections.size(); ++j) {
            if(gating_distance(j)>gating_threshold) {
                cost_matrix(i,j) = kMaxCost;
            } else {
                cost_matrix(i,j) = lambda*cost_matrix(i,j)+(1.0f-lambda)*gating_distance(j);
            }
        }
    }
}
}
