#include "kalman_filter.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <stdexcept>
#include <iostream>

using namespace std;
using namespace Eigen;
using namespace MOT;

KalmanFilter::KalmanFilter()
{
    /*
     * Create Kalman filter model matrices.
     * this set of _motion_mat mean x_k = x_(k-1)+v_x,...,v_x_k = v_x_(k-1)
     * motion_mat_ is F_k
     */
    motion_mat_.setIdentity();
    motion_mat_.block<4,4>(0,4).setIdentity();
    /*
     * this set of _update_mat mean mean_expected = mean_k, covariance_expected = zero
     * update_mat is inv(H_k) in paper, project state to measure, (in paper project measure to state).
     */
    update_mat_.setIdentity();
}
MeanCov_t KalmanFilter::initiate(const Eigen::VectorXf& measurement)
{
        /*
        Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        */
        Mean_t mean;
        mean.setZero();
        mean.block<4,1>(0,0) = measurement;

        VectorXf std(8);
        std[0] = 2*std_weight_position_*measurement[3];
        std[1] = 2*std_weight_position_*measurement[3];
        std[2] = 1e-2;
        std[3] = 2*std_weight_position_*measurement[3];
        std[4] = 10*std_weight_velocity_*measurement[3];
        std[5] = 10*std_weight_velocity_*measurement[3];
        std[6] = 1e-5;
        std[7] = 10*std_weight_velocity_*measurement[3];

        Cov_t covariance = Cov_t(std.asDiagonal()).array().pow(2);

        return make_pair(mean,covariance);
}
MeanCov_t KalmanFilter::predict(const Mean_t& mean,const Cov_t& covariance)
{
    /*
         Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

     */
        VectorXf std(8);
        std[0] = std_weight_position_*mean[3];
        std[1] = std_weight_position_*mean[3];
        std[2] = 1e-2;
        std[3] = std_weight_position_*mean[3];
        std[4] = std_weight_velocity_*mean[3];
        std[5] = std_weight_velocity_*mean[3];
        std[6] = 1e-5;
        std[7] = std_weight_velocity_*mean[3];

        //motion_conv is Q_k (external noise)
        Cov_t motion_cov = Cov_t(std.asDiagonal()).array().square();
        Mean_t r_mean = motion_mat_*mean;

        Cov_t r_covariance = motion_mat_*covariance*motion_mat_.transpose()+motion_cov;
        return make_pair(r_mean,r_covariance);
}
PMeanCov_t KalmanFilter::project(const Mean_t& mean,const Cov_t& covariance)
{
    /*
        Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
            which is:H_k X_k, H_k P_k H_k^T +R_k
     */
        VectorXf std(4);
        std[0] = std_weight_position_*mean[3];
        std[1] = std_weight_position_*mean[3];
        std[2] = 1e-1;
        std[3] = std_weight_position_*mean[3];

        //innovation_cov is muasurement noise R_k
        PCov_t innovation_cov = PCov_t(std.asDiagonal()).array().square();
        PMean_t r_mean = update_mat_*mean;
        auto r_cov = update_mat_*covariance*update_mat_.transpose()+innovation_cov;

        return make_pair(r_mean,r_cov);
}
MeanCov_t KalmanFilter::update(const Mean_t& mean,const Cov_t& covariance,const Eigen::VectorXf& measurement)
{
    /*
        Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
     */
     PMean_t projected_mean;
     PCov_t projected_cov;

     tie(projected_mean,projected_cov) = project(mean,covariance);

     using TD = Matrix<float,kNDim,2*kNDim>;

     TD b = update_mat_*covariance.transpose();
     auto kgt = solve<TD>(projected_cov,b);

     Matrix<float,2*kNDim,kNDim> kalman_gain = kgt.transpose();
     
    PMean_t innovation = measurement-projected_mean;
    Mean_t new_mean = mean+kalman_gain*innovation;
    //K PC K^T = K H_k P_k
    Matrix<float,2*kNDim,kNDim> left_kp = kalman_gain*projected_cov;
    Cov_t new_covariance = covariance-left_kp*kgt;

    return make_pair(new_mean,new_covariance); 
}
Eigen::VectorXf KalmanFilter::gating_distance(const Mean_t& mean,const Cov_t& covariance,const Eigen::MatrixXf& measurement,bool only_position,Metric metric)
{
    /*
        Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
     */
    VectorXf projected_mean;
    MatrixXf projected_cov;
    auto l_measurement = measurement;

    tie(projected_mean,projected_cov) = project(mean,covariance);

    if(only_position) {
        projected_mean = projected_mean.block<2,1>(0,0);
        projected_cov = projected_cov.block<2,2>(0,0);
        l_measurement = l_measurement.block(0,0,l_measurement.rows(),2);
    }

    auto d = l_measurement.rowwise()-projected_mean.transpose();

    if(GAUSSIAN == metric) {
        return d.array().pow(2).rowwise().sum();
    } else if(MAHA == metric) {
        MatrixXf L = projected_cov.llt().matrixL();
        auto z = solve<MatrixXf>(L,(MatrixXf)d.transpose());
        return z.transpose().array().square().rowwise().sum();
    } else {
        throw std::runtime_error("Error metric type.");
    }
}
