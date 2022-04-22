#ifndef MONO_TRACKING_MOTION_FILTER_HPP
#define MONO_TRACKING_MOTION_FILTER_HPP

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/video/tracking.hpp>

#include <mono_tracking/track_system.hpp>

namespace mono_tracking
{

class KalmanFilter
{
public:
    KalmanFilter(const std::shared_ptr<TrackSystem>& track_system, int dynamParams, int measureParams, int controlParams, const Eigen::Matrix<float, 4, 1>& mean, const Eigen::Matrix<float, 4, 4>& cov)
    :   system(track_system), mean(mean), cov(cov)
    {
        K = system->camera_matrix;
        // Tcr = system->footprint2camera.matrix();
        // Trw = system->odom2footprint.matrix();
        Tcr = system->base2camera.matrix();
        Trw = system->odom2base.matrix();
        process_noise = system->process_noise;
        measurement_noise = system->measurement_noise;

        init_transition_matrix(system->dt);
        init_measurement_matrix();
        // cout << "Transition Matrix" << endl << transition_matrix << endl;
        // cout << "Measurement Matrix" << endl << measurement_matrix << endl;
        // cout << "mean" << endl << mean << endl;
        // cout << "cov" << endl << cov << endl;
        
        // init kalman filter
        kf = new cv::KalmanFilter(dynamParams, measureParams, controlParams);
        cv::eigen2cv(transition_matrix, kf->transitionMatrix);
        cv::eigen2cv(measurement_matrix, kf->measurementMatrix);
        cv::eigen2cv(process_noise, kf->processNoiseCov);
        cv::eigen2cv(measurement_noise, kf->measurementNoiseCov);
        cv::setIdentity(kf->errorCovPost, cv::Scalar::all(1));
    }
    Eigen::VectorXf predict()
    {
        update_transition_matrix(system->dt);
        // cout << "Before Predict mean: " << endl << mean << endl;
        // cout << "Transition Matrix" << endl << transition_matrix << endl;
        cv::eigen2cv(mean, kf->statePost);
        cv::eigen2cv(transition_matrix, kf->transitionMatrix);
        // Eigen::Vector3f prediction;
        kf->predict();
        cv::cv2eigen(kf->statePost, mean);
        cv::cv2eigen(kf->errorCovPost, cov);
        // cout << "mean: " << endl << mean << endl;
        // cout << "cov: " << endl << cov << endl;
        return mean;
    }
    Eigen::VectorXf update(const Eigen::Vector2f& y_)
    {
        // cout << "Measurement Matrix" << endl << measurement_matrix << endl;
        cv::eigen2cv(measurement_matrix, kf->measurementMatrix);
        // Eigen::Vector3f estimated;
        cv::Mat observation;
        cv::eigen2cv(y_, observation);
        kf->correct(observation);
        cv::cv2eigen(kf->statePost, mean);  // kf->correct() will not change statePre, but statePost
        cv::cv2eigen(kf->errorCovPost, cov);
        
        return mean;
    }
    std::pair<Eigen::Matrix<float, 2, 1>, Eigen::Matrix<float, 2, 2>> expected_measurement_distribution()
    {
        // y = Hx + measurement_cov -> mean'=H*mean, cov'=H*cov*Ht + measurement_cov
        Eigen::VectorXf expected_measurement_mean = measurement_matrix*mean;
        // cout << "measurement_matrix" << endl << measurement_matrix << endl;
        // cout << "cov" << endl << cov << endl;
        // cout << "measurement_noise" << endl << measurement_noise << endl;
        Eigen::MatrixXf expected_measurement_cov = measurement_matrix*cov*measurement_matrix.transpose() + measurement_noise;
        return std::make_pair(expected_measurement_mean, expected_measurement_cov);
    }

    // For kalman filter parameters
    void init_transition_matrix(double _dt)
    {
        // constant velocity model
        transition_matrix.setIdentity();
        transition_matrix(0,2) = _dt;
        transition_matrix(1,3) = _dt;
    }

    void update_transition_matrix(double _dt)
    {
        transition_matrix(0,2) = _dt;
        transition_matrix(1,3) = _dt;
    }

    void init_measurement_matrix()
    {
        measurement_matrix.setZero();
        Eigen::Matrix3f Rcw = Tcr.block<3,3>(0,0)*Trw.block<3,3>(0,0);
        measurement_matrix.block<1,2>(0,0) = Rcw.block<1,2>(0,0);
        measurement_matrix.block<1,2>(1,0) = Rcw.block<1,2>(2,0);
    }

public:
    // kalman filter
    cv::KalmanFilter* kf;

    // state
    Eigen::Matrix<float, 4, 1> mean;  // [px, py, vx, vy]
    Eigen::Matrix<float, 4, 4> cov;  // identity([1,1,1,1])

    // matrix
    Eigen::Matrix3f K;
    // Eigen::Matrix<float, 3, 4> expanded_K;
    Eigen::Matrix4f Tcr, Trw;
    // Eigen::Matrix<float, 4, 6> expanded_Trw;

    Eigen::Matrix<float, 4, 4> transition_matrix;
    Eigen::Matrix<float, 2, 4> measurement_matrix;
    Eigen::Matrix<float, 4, 4> process_noise;
    Eigen::Matrix<float, 2, 2> measurement_noise;

    // track system
    std::shared_ptr<TrackSystem> system;
};
}


#endif // MOTION_FILTER_HPP