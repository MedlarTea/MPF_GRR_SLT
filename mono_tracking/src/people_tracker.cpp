#include <mono_tracking/people_tracker.hpp>

#include <kkl/alg/data_association.hpp>
#include <kkl/alg/nearest_neighbor_association.hpp>

#include <mono_tracking/track_system.hpp>

namespace mono_tracking
{
PeopleTracker::PeopleTracker(ros::NodeHandle& private_nh, const std::shared_ptr<TrackSystem>& _track_system, const std::shared_ptr<tf::TransformListener>& tf_listener)
{
    id_gen = 0;
    remove_trace_thresh = private_nh.param<double>("tracking_remove_trace_thresh", 5.0);
    dist_to_exists_thresh = private_nh.param<double>("tracking_newtrack_dist2exists_thersh", 100.0);

    data_association.reset(new kkl::alg::NearestNeighborAssociation<PersonTracker::Ptr, Observation::Ptr, AssociationDistance>(AssociationDistance(private_nh)));
    // track_system.reset(new TrackSystem(private_nh, tf_listener, camera_frame_id, camera_info_msg));
    track_system = _track_system;
}

PeopleTracker::~PeopleTracker() {}

// TODO: Use Multi-thread to speed up
// deltaT may need to be locked
void PeopleTracker::predict(ros::NodeHandle& nh, const ros::Time& stamp) {
    track_system->update_matrices(stamp);
    for(const auto& person : people) {
        // cout << "Person PREDICT" << endl;
        // cout << "----------" << endl;
        // cout << "ID: " << person->id() << endl;
        // cout << "PREVIOUS" << endl;
        // cout << "Position" << endl << person->pos() << endl;
        // cout << "Vel" << endl << person->vel() << endl;
        // cout << "Cov" << endl << person->cov() << endl;
        person->predict(stamp);
        // cout << "AFTER" << endl;
        // cout << "Position" << endl << person->pos() << endl;
        // cout << "Vel" << endl << person->vel() << endl;
        // cout << "Cov" << endl << person->cov() << endl;
        // cout << "----------" << endl;

    }
}

void PeopleTracker::update(ros::NodeHandle& nh, const ros::Time& stamp, const std::vector<Observation::Ptr>& observations) {
    if(!observations.empty()) {

        std::vector<bool> associated(observations.size(), false);
        auto associations = data_association->associate(people, observations);
        // cout << "people nums: " << people.size() << endl;
        // cout << "obs nums: " << observations.size() << endl;
        
        for(const auto& assoc : associations) {
            // cout << "ID: " << people[assoc.tracker]->id() << endl;
            associated[assoc.observation] = true;
            people[assoc.tracker]->update(observations[assoc.observation]);
            // cout << "Person UPDATE" << endl;
            // cout << "----------" << endl;
            // cout << "Position" << endl << people[assoc.tracker]->pos() << endl;
            // cout << "Cov" << endl << people[assoc.tracker]->cov() << endl;
            // cout << "----------" << endl;
        }

        // observations without matching tracker
        for(int i=0; i<observations.size(); i++) {
            if(!associated[i]) {
                cout << "Person INIT" << endl;
                cout << "----------" << endl;
                PersonTracker::Ptr tracker(new PersonTracker(nh, track_system, stamp, id_gen++, observations[i]->obs));
                tracker->update(observations[i]);  // Q: Why we need to update the tracker after initialized?? Iw will cause the state cov is ZERO!!!
                                                   // A: Decrease the covariance
                people.push_back(tracker);
                cout << "----------" << endl;
            }
        }
    }

    // state cov trace is removing standard
    auto remove_loc = std::partition(people.begin(), people.end(), [&](const PersonTracker::Ptr& tracker) {
        return tracker->trace() < remove_trace_thresh;
    });
    removed_people.clear();
    std::copy(remove_loc, people.end(), std::back_inserter(removed_people));
    people.erase(remove_loc, people.end());
}


}