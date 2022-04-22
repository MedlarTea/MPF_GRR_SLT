#ifndef MONO_TRACKING_ASSOCIATION_DISTANCE_HPP
#define MONO_TRACKING_ASSOCIATION_DISTANCE_HPP

#include <memory>
#include <vector>
#include <boost/optional.hpp>

#include <tf/transform_listener.h>
#include <sensor_msgs/CameraInfo.h>
#include <mono_tracking/person_tracker.hpp>


namespace mono_tracking
{

class AssociationDistance
{
public:
    AssociationDistance(ros::NodeHandle& private_nh)
      : maha_sq_thresh(private_nh.param<double>("association_maha_sq_thresh", 9.0)),
        max_dist(private_nh.param<int>("max_dist", 200)),
        lambda(private_nh.param<double>("lambda", 0.004))
    {

    }

    boost::optional<double> operator() (const PersonTracker::Ptr& tracker, const Observation::Ptr& observation) const
    {
        auto expected_measurement = tracker->expected_measurement_distribution();
        double expected_obs0 = expected_measurement.first(0);  // u
        double expected_obs1 = expected_measurement.first(1);  // distance

        double distance = sqrt(pow((expected_obs0-observation->obs[0]), 2) + pow((expected_obs1-observation->obs[1]), 2));
        // double distance = abs(expected_obs0-observation->obs[0]);
        // double distance = tracker->squared_mahalanobis_distance(observation->obs);
        if(distance>max_dist)
            return boost::none;
        // cout << "distance: " << distance << endl;
        return distance;
    }


private:
    double lambda;
    double maha_sq_thresh;
    int max_dist;

};

}

#endif
