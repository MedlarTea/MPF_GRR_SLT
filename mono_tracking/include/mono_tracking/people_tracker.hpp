#ifndef MONO_TRACKING_PEOPLE_TRACKER_HPP
#define MONO_TRACKING_PEOPLE_TRACKER_HPP

#include <memory>
#include <vector>
#include <boost/optional.hpp>

#include <tf/transform_listener.h>
#include <sensor_msgs/CameraInfo.h>
#include <mono_tracking/person_tracker.hpp>
#include <mono_tracking/association_distance.hpp>


namespace kkl {
  namespace alg {
    template<typename Tracker, typename Observation>
    class DataAssociation;
  }
}


namespace mono_tracking
{

class TrackSystem;

class PeopleTracker
{
public:
    PeopleTracker(ros::NodeHandle& private_nh, const std::shared_ptr<TrackSystem>& _track_system, const std::shared_ptr<tf::TransformListener>& tf_listener);
    ~PeopleTracker();

    void predict(ros::NodeHandle& nh, const ros::Time& stamp);

    void update(ros::NodeHandle& nh, const ros::Time& stamp, const std::vector<Observation::Ptr>& observations);

    const std::vector<PersonTracker::Ptr>& get_people() const { return people; }

private:
    std::shared_ptr<TrackSystem> track_system;
    std::unique_ptr<kkl::alg::DataAssociation<PersonTracker::Ptr, Observation::Ptr>> data_association;

    double remove_trace_thresh;
    double dist_to_exists_thresh;

    long id_gen;
    std::vector<PersonTracker::Ptr> people;
    std::vector<PersonTracker::Ptr> removed_people;
};

}

#endif // PEOPLE_TRACKER_HPP
