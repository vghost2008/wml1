#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

std::vector<std::vector<std::pair<float,float>>> openpose_decode_imp(const cv::Mat& conf_map,const cv::Mat& paf_map,
        const std::vector<std::pair<int,int>>& map_idx,
        const std::vector<std::pair<int,int>>& pose_pairs,
        const float keypoints_th=0.1,
        const int interp_samples=10,
        const float paf_score_th=0.1,
        const float conf_th=0.7);
