#ifndef MONO_TRACKING_TRACK_SYSTEM_HPP
#define MONO_TRACKING_TRACK_SYSTEM_HPP

#include <ros/node_handle.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/CameraInfo.h>

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>

namespace mono_tracking
{
class TrackSystem
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    TrackSystem(ros::NodeHandle& private_nh, const std::shared_ptr<tf::TransformListener>& tf_listener, const std::string& camera_frame_id, const sensor_msgs::CameraInfoConstPtr& camera_info_msg)
        : camera_frame_id(camera_frame_id),
        tf_listener(tf_listener)
    {
        dt = 0.1;
        // Linear regression
        // k = private_nh.param<float>("coefficient_distance_width", -48.494);
        // b = private_nh.param<float>("bias_distance_width", 349.495);

        // Exponential regression
        // k = private_nh.param<float>("coefficient_width_distance", -0.0062);
        // b = private_nh.param<float>("bias_width_distance", 2.2757);

        measurement_noise = Eigen::Matrix2f::Identity();
        measurement_noise(0,0) =  private_nh.param<double>("measurement_noise_distance_cov", 0.1);
        measurement_noise(1,1) =  private_nh.param<double>("measurement_noise_distance_cov", 0.1);
        
        process_noise.setIdentity();
        process_noise.middleRows(0, 2) *= private_nh.param<double>("process_noise_pos_cov", 0.1);
        // process_noise.middleRows(2, 2) *= 0;  // suppose person's height is fixed and the fourth element is 1
        process_noise.middleRows(2, 2) *= private_nh.param<double>("process_noise_vel_cov", 0.1);

        camera_matrix = Eigen::Map<const Eigen::Matrix3d>(camera_info_msg->K.data()).transpose().cast<float>();
        update_matrices(ros::Time(0));
    }
    
    void update_matrices(const ros::Time& stamp) {
        
        base2camera = lookup_eigen(camera_frame_id, "base_link", stamp);
        // odom2camera = lookup_eigen(camera_frame_id, "odom", stamp);
        // odom2base = lookup_eigen("base_link", "odom", stamp);
        odom2base = Eigen::Isometry3f::Identity();
        odom2camera = odom2base * base2camera;
    }

    Eigen::Isometry3f lookup_eigen(const std::string& to, const std::string& from, const ros::Time& stamp) {
        tf::StampedTransform transform;
        try{
        tf_listener->waitForTransform(to, from, stamp, ros::Duration(1.0));
        tf_listener->lookupTransform(to, from, stamp, transform);
        } catch (tf::ExtrapolationException& e) {
        tf_listener->waitForTransform(to, from, ros::Time(0), ros::Duration(5.0));
        tf_listener->lookupTransform(to, from, ros::Time(0), transform);
        }

        Eigen::Isometry3d iso;
        tf::transformTFToEigen(transform, iso);
        return iso.cast<float>();
    }

    Eigen::Vector3f transform_odom2camera(const Eigen::Vector3f& pos_in_odom) const {
        return (odom2camera * Eigen::Vector4f(pos_in_odom.x(), pos_in_odom.y(), pos_in_odom.z(), 1.0f)).head<3>();
    }

    // Eigen::Vector3f transform_odom2footprint(const Eigen::Vector3f& pos_in_odom) const {
    //     return (odom2footprint * Eigen::Vector4f(pos_in_odom.x(), pos_in_odom.y(), pos_in_odom.z(), 1.0f)).head<3>();
    // }

    // Eigen::Vector3f transform_footprint2odom(const Eigen::Vector3f& pos_in_footprint) const {
    //     return (odom2footprint.inverse() * Eigen::Vector4f(pos_in_footprint.x(), pos_in_footprint.y(), pos_in_footprint.z(), 1.0f)).head<3>();
    // }

    void set_dt(double d) {
        dt = std::max(d, 1e-9);
    }

    Eigen::MatrixXf processNoiseCov() const {
        return process_noise;
    }

    template<typename Measurement>
    Measurement h(const Eigen::VectorXf& state) const;

    template<typename Measurement>
    Eigen::MatrixXf measurementNoiseCov() const;

public:
    Eigen::Isometry3f odom2camera;
    // Eigen::Isometry3f odom2footprint;
    // Eigen::Isometry3f footprint2base;
    // Eigen::Isometry3f footprint2camera;
    Eigen::Isometry3f base2camera;
    Eigen::Isometry3f odom2base;

    Eigen::Matrix3f camera_matrix;

    std::string camera_frame_id;
    std::shared_ptr<tf::TransformListener> tf_listener;

    // Filter parameters
    double dt;  // delta t
    Eigen::Matrix<float, 4, 4> process_noise;
    Eigen::Matrix2f measurement_noise;

    // coeff and bias, represent width-distance
    double k,b;

};

}



#endif // PERSON_TRACKER_HPP