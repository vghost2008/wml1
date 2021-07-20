#pragma once
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <utility>
#include <memory>

/*
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a (w/h), height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
*/
namespace MOT
{
    constexpr float chi2inv95[] = { 3.8415, 5.9915, 7.8147, 9.4877, 11.070, 12.592, 14.067, 15.507, 16.919};
    constexpr auto kNDim = 4;
    constexpr auto kDt = 1.0f;
    using Cov_t = Eigen::Matrix<float,kNDim*2,kNDim*2>;
    using Mean_t = Eigen::Matrix<float,kNDim*2,1>;
    using MeanCov_t = std::pair<Mean_t,Cov_t>;
    using PCov_t = Eigen::Matrix<float,kNDim,kNDim>;
    using PMean_t = Eigen::Matrix<float,kNDim,1>;
    using PMeanCov_t = std::pair<PMean_t,PCov_t>;
    class KalmanFilter
    {
        public:
            enum Metric{
                MAHA,
                GAUSSIAN,
            };
        public:
            KalmanFilter();
            void update_features(const Eigen::VectorXf& feat);
            MeanCov_t initiate(const Eigen::VectorXf& measurement);
            MeanCov_t predict(const Mean_t& mean,const Cov_t& covariance);
            PMeanCov_t project(const Mean_t& mean,const Cov_t& covariance);
            MeanCov_t update(const Mean_t& mean,const Cov_t& covariance,const Eigen::VectorXf& measurement);
            Eigen::VectorXf gating_distance(const Mean_t& mean,const Cov_t& covariance,const Eigen::MatrixXf& measurement,bool only_position=false,Metric metric=MAHA);
        private:
            template<typename T0,typename T1,typename T2>
                T0 solve(const T1& A,const T2& b) {
                    return A.llt().solve(b);
                }
        private:
            Eigen::Matrix<float,kNDim*2,kNDim*2> motion_mat_;  //F_k
            Eigen::Matrix<float,kNDim,2*kNDim> update_mat_; //inv(H_k)
            static constexpr auto std_weight_position_ = 1.0f/20.0f;
            static constexpr auto std_weight_velocity_ = 1.0f/160.0f;
    };
    using KalmanFilterPtr_t = std::shared_ptr<KalmanFilter>;
}
