#include <mono_tracking/person_tracker.hpp>

#include <cppoptlib/problem.h>
#include <cppoptlib/solver/neldermeadsolver.h>

namespace mono_tracking
{
PersonTracker::PersonTracker(ros::NodeHandle& nh, const std::shared_ptr<TrackSystem>& track_system, const ros::Time& stamp, long id, const Eigen::Vector2f& center_width)
: id_(id) 
{   
    Eigen::VectorXf mean = Eigen::VectorXf::Zero(4);  // [px, py, vx, vy]
    mean.head<3>() = estimate_init_state(track_system, center_width);
    cout << "Obs: " << endl;
    cout << center_width << endl;
    cout << "Person init state: " << endl;
    cout << mean.head<2>() << endl;
    Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(4, 4) * nh.param<double>("init_cov_scale", 1.0);

    kf.reset(new KalmanFilter(track_system, 4, 2, 0, mean, cov));

    prev_stamp = stamp;
    correction_count_ = 0;
    validation_correction_count = 2;
    last_associated = nullptr;
}
PersonTracker::~PersonTracker(){}

// Motion filter
Eigen::VectorXf PersonTracker::predict(const ros::Time& stamp)
{
    expected_measurement_dist = boost::none;
    last_associated = nullptr;
    kf->system->set_dt((stamp - prev_stamp).toSec());
    prev_stamp = stamp;
    return kf->predict();
}

Eigen::VectorXf PersonTracker::update(const Observation::Ptr& observation)
{
    expected_measurement_dist = boost::none;
    last_associated = observation;
    correction_count_ ++;
    return kf->update(observation->obs);
}

// Get attributes
std::pair<Eigen::Matrix<float, 2, 1>, Eigen::Matrix<float, 2, 2>> PersonTracker::expected_measurement_distribution() const
{
    if(!expected_measurement_dist)
        expected_measurement_dist = kf->expected_measurement_distribution();

    return *expected_measurement_dist;
}

double PersonTracker::trace() const {
  return kf->cov.trace();
}

Eigen::Vector2f PersonTracker::pos() const {
  return kf->mean.head<2>();
}

Eigen::Vector2f PersonTracker::vel() const {
  return kf->mean.tail<2>();
}

Eigen::MatrixXf PersonTracker::cov() const {
  return kf->cov;
}

// METHOD2 BY ANALYZATION
Eigen::Vector3f PersonTracker::estimate_init_state(const std::shared_ptr<TrackSystem>& track_system, const Eigen::Vector2f& center_width) const{
    // cout << "track_system->camera_matrix " << endl << track_system->camera_matrix << endl;
    Eigen::Matrix4f Tcr,Trw;
    // Tcr = track_system->footprint2camera.matrix();
    // Trw = track_system->odom2footprint.matrix();
    Tcr = track_system->base2camera.matrix();
    Trw = track_system->odom2base.matrix();
    Eigen::Matrix3f Rcw = Tcr.block<3,3>(0,0)*Trw.block<3,3>(0,0);
    Eigen::Matrix2f A;
    A.block<1,2>(0,0) = Rcw.block<1,2>(0,0);
    A.block<1,2>(1,0) = Rcw.block<1,2>(2,0);
    Eigen::Vector3f state;
    state.setZero();
    state.head<2>() = A.inverse()*center_width;
    return state;
}

}