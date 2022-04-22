#ifndef MONO_TRACKING_PERSON_TRACKER_HPP
#define MONO_TRACKING_PERSON_TRACKER_HPP

#include <ros/ros.h>
#include <memory>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <kkl/math/gaussian.hpp>
#include <mono_tracking/observation.hpp>
#include <mono_tracking/track_system.hpp>
#include <mono_tracking/motion_filter.hpp>

using namespace std;



namespace mono_tracking
{

class PersonTracker{

public:
    using Ptr = std::shared_ptr<PersonTracker>;
    PersonTracker(ros::NodeHandle& nh, const std::shared_ptr<TrackSystem>& track_system, const ros::Time& stamp, long id, const Eigen::Vector2f& center_width);
    ~PersonTracker();

    Eigen::VectorXf predict(const ros::Time& stamp);
    Eigen::VectorXf update(const Observation::Ptr& observation);

    double trace() const;
    Eigen::Vector2f pos() const;  // homogeneous coordinates
    Eigen::Vector2f vel() const;  // vx vy
    Eigen::MatrixXf cov() const;  // state covariance
    // cv::Mat descriptor;  // newest descriptor
    // vector<cv::Mat> descriptorCache;  // descriptors cache

    Observation::Ptr last_associated;

    long id() const {
        return id_;
    }

    long correction_count() const {
        return correction_count_;
    }

    bool is_valid() const {
        return correction_count() > validation_correction_count;
    }

    

public:
    double squared_mahalanobis_distance(const Eigen::Vector2f& x) const {
        std::pair<Eigen::Matrix<float, 2, 1>, Eigen::Matrix<float, 2, 2>> dist = expected_measurement_distribution();
        Eigen::Vector2f mean = dist.first.head<2>();
        Eigen::Matrix2f cov = dist.second.block<2, 2>(0, 0);
        return kkl::math::squaredMahalanobisDistance<float, 2>(mean, cov, x);
    }
    std::pair<Eigen::Matrix<float, 2, 1>, Eigen::Matrix<float, 2, 2>> expected_measurement_distribution() const;


private:
    Eigen::Vector3f estimate_init_state(const std::shared_ptr<TrackSystem>& track_system, const Eigen::Vector2f& center_width) const;
private:
    long id_;
    long correction_count_;
    long validation_correction_count;
    ros::Time prev_stamp;

    
    std::unique_ptr<KalmanFilter> kf;  // for state predict and update
    mutable boost::optional<std::pair<Eigen::Matrix<float, 2, 1>, Eigen::Matrix<float, 2, 2>>> expected_measurement_dist;
};  



} // namespace mono_tracking
#endif // PERSON_TRACKER_HPP



