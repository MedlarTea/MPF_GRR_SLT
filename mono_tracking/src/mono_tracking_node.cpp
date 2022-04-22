#include <memory>
#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <boost/format.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <visualization_msgs/MarkerArray.h>
#include <mono_tracking/Box.h>
#include <mono_tracking/BoxArray.h>
#include <mono_tracking/Track.h>
#include <mono_tracking/TrackArray.h>

#include <kkl/cvk/cvutils.hpp>
#include <kkl/math/gaussian.hpp>

#include <mono_tracking/observation.hpp>
#include <mono_tracking/people_tracker.hpp>

using namespace std::chrono;
using namespace std;

namespace mono_tracking
{
class MonoTrackingNode
{
public:
    MonoTrackingNode()
    : nh(),
      private_nh("~"),
      poses_sub(nh.subscribe("/people/boxes", 10, &MonoTrackingNode::boxes_callback, this)),
      camera_info_sub(nh.subscribe("/camera/color/camera_info", 60, &MonoTrackingNode::camera_info_callback, this)),
      tracks_pub(private_nh.advertise<mono_tracking::TrackArray>("tracks", 10)),
      markers_pub(private_nh.advertise<visualization_msgs::MarkerArray>("markers", 10)),
      image_trans(private_nh),
      image_pub(image_trans.advertise("tracking_image", 5))
    {
        ROS_INFO("Start Mono_Tracking_Node");
        color_palette = cvk::create_color_palette(16);
        tf_listener.reset(new tf::TransformListener());
    }



private:
    void camera_info_callback(const sensor_msgs::CameraInfoConstPtr& _camera_info_msg) 
    {   
        if(camera_info_msg == nullptr)
        {
            ROS_INFO("camera_info");
            this->camera_info_msg = _camera_info_msg;
        }
    }

    bool check_IoU(vector<u_int16_t> box1_xywh, vector<u_int16_t> box2_xywh, const sensor_msgs::CameraInfoConstPtr& camera_info_msg){
        float IOU_THRESHOLD = 0.5;
        vector<u_int16_t> box1, box2;  // x1,y1,x2,y2
        box1.push_back(max(int(box1_xywh[0]-box1_xywh[2]/2),0));
        box1.push_back(max(int(box1_xywh[1]-box1_xywh[3]/2),0));
        box1.push_back(min(int(box1_xywh[0]+box1_xywh[2]/2),int(camera_info_msg->width-1)));
        box1.push_back(min(int(box1_xywh[1]+box1_xywh[3]/2),int(camera_info_msg->height-1)));

        box2.push_back(max(int(box2_xywh[0]-box2_xywh[2]/2),0));
        box2.push_back(max(int(box2_xywh[1]-box2_xywh[3]/2),0));
        box2.push_back(min(int(box2_xywh[0]+box2_xywh[2]/2),int(camera_info_msg->width-1)));
        box2.push_back(min(int(box2_xywh[1]+box2_xywh[3]/2),int(camera_info_msg->height-1)));

        u_int16_t xA = std::max(box1[0], box2[0]);
        u_int16_t yA = std::max(box1[1], box2[1]);
        u_int16_t xB = std::min(box1[2], box2[2]);
        u_int16_t yB = std::min(box1[3], box2[3]);

        u_int32_t interArea = max(0, xB-xA+1) * max(0, yB-yA+1);
        u_int32_t boxAArea = (box1[2]-box1[0]+1) * (box1[3]-box1[1]+1);
        u_int32_t boxBArea = (box2[2]-box2[0]+1) * (box2[3]-box2[1]+1);

        float iou = interArea / float(boxAArea+boxBArea-interArea);
        return iou > IOU_THRESHOLD;
    }

    vector<bool> boxesCheck(const mono_tracking::BoxArrayConstPtr& boxes_msg, const sensor_msgs::CameraInfoConstPtr& camera_info_msg){
        vector<bool> flags(boxes_msg->boxes.size(), true);
        // cout << "boxes: " << boxes_msg->boxes.size() << endl;
        // cout << "flags: " << flags.size() << endl;
        if(flags.size()<=1){
            return flags;
        }
        for(int i=0; i<(boxes_msg->boxes.size()-1); i++){
            for(int j=i+1; j<(boxes_msg->boxes.size()); j++){
                if(check_IoU(boxes_msg->boxes[i].box, boxes_msg->boxes[j].box, camera_info_msg)){
                    flags[i] = flags[j] = false;
                }
            }
        }
        // cout << "flags: " << flags.size() << endl;
        return flags;
    }

    void boxes_callback(const mono_tracking::BoxArrayConstPtr& boxes_msg)
    {
        // auto start = high_resolution_clock::now();
        // ROS_INFO("Boxes_Callback");
        if(camera_info_msg == nullptr)
        {
            ROS_INFO("waiting for the camera info msg...");
            return;
        }
        if(!track_system)
        {
            track_system.reset(new TrackSystem(private_nh, tf_listener, boxes_msg->header.frame_id, camera_info_msg));
            return;
        }
        // cout << boxes_msg->boxes.size() << endl;
        vector<bool> goodBoxes = boxesCheck(boxes_msg, camera_info_msg);
        vector<mono_tracking::Box> boxes;
        for (int i = 0; i < goodBoxes.size();i++){
            if(goodBoxes[i]==true){
                boxes.push_back(boxes_msg->boxes[i]);
            }
        }
        // if (boxes.size()==0){
        //     // ROS_INFO("Bad observations");
        //     // return;
        // }

        std::vector<Observation::Ptr> observations;
        
        observations.reserve(boxes.size());
        // cout << "peopleNums: " << people_tracker->get_people().size() << endl;
        for(const auto& box:boxes){
            auto observation = std::make_shared<Observation>(private_nh, track_system, camera_info_msg, box.box, box.score);
            if(observation->isGoodObservation){
                observations.push_back(observation);
            }
        }

        if(!people_tracker){
            cout << "People Init" << endl;
            people_tracker.reset(new PeopleTracker(private_nh, track_system, tf_listener));
        }
        
        // update the tracker
        const auto& stamp = boxes_msg->header.stamp;
        // cout << "People Predict" << endl;
        people_tracker->predict(private_nh, stamp);
        // cout << "People Update" << endl;
        people_tracker->update(private_nh, stamp, observations);

        // publish tracks
        if(tracks_pub.getNumSubscribers())
            tracks_pub.publish(create_people_msgs(boxes_msg->header.stamp, boxes_msg->image));
        
        // publish visualization msgs
        if(image_pub.getNumSubscribers()){
            // Convert sensor_msgs::Image to cv::Mat
            cv_bridge::CvImagePtr cv_ptr;
            try{
                cv_ptr = cv_bridge::toCvCopy(boxes_msg->image, sensor_msgs::image_encodings::BGR8);
            }
            catch (cv_bridge::Exception& e){
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }
            cv::Mat frame = cv_ptr->image;
            cout << "People Visualizing" << endl;
            visualize(frame, observations);
            cv_ptr->encoding = "bgr8";
            cv_ptr->header.stamp = ros::Time::now();
            cv_ptr->image = frame;
            image_pub.publish(cv_ptr->toImageMsg());
            cout << "People Sending" << endl;
        }

        // publish markers
        if(markers_pub.getNumSubscribers())
            markers_pub.publish(create_markers(boxes_msg->header.stamp));
        
        // auto stop = high_resolution_clock::now();
        // auto duration = duration_cast<milliseconds>(stop - start);
        // cout << "DURATION: " << duration.count() << "ms" << endl;
    }

    mono_tracking::TrackArrayPtr create_people_msgs(const ros::Time& stamp, sensor_msgs::Image _image)
    {
        mono_tracking::TrackArrayPtr msgs(new mono_tracking::TrackArray());
        if(!people_tracker)
            return msgs;
        msgs->header.stamp = stamp;
        msgs->header.frame_id = "odom";
        msgs->image = _image;
        // cout << "peopleNums: " << people_tracker->get_people().size() << endl;
        for(const auto& person : people_tracker->get_people())
        {
            if(!person->is_valid())
                continue;
            auto& tracks = msgs->tracks;
            mono_tracking::Track tr;
            Eigen::Vector2f pos = person->pos();
            Eigen::Vector2f vel = person->vel();
            tr.id = person->id();
            // cout << "id: " << person->id() << endl;
            // tr.age = 0;
            tr.pos.x = pos.x();
            tr.pos.y = pos.y();
            tr.pos.z = 0;
            tr.vel.x = vel.x();
            tr.vel.y = vel.y();
            tr.vel.z = 0;
            tr.trace = person->trace();

            if(person->last_associated!=nullptr){
                tr.last_associated[0] = person->last_associated->obs(0);
                tr.last_associated[1] = person->last_associated->obs(1);
                mono_tracking::Box box;
                box.box.push_back(person->last_associated->x1);
                box.box.push_back(person->last_associated->y1);
                box.box.push_back(person->last_associated->x2);
                box.box.push_back(person->last_associated->y2);
                tr.box = box;
                tr.box.score = person->last_associated->score;
            }

            for(auto i = 0; i < person->cov().size(); i++) 
            {
                tr.cov[i] = person->cov().array()(i);
            }
            // cout << "id: " << person->id() << endl;
            auto dist = person->expected_measurement_distribution();
            for(size_t i=0; i<dist.first.size(); i++) 
            {
                tr.expected_measurement_mean[i] = dist.first[i];
            }
            for(size_t i=0; i<dist.second.rows(); i++) 
            {
                for(size_t j=0; j<dist.second.cols(); j++) 
                {
                    tr.expected_measurement_cov[i * dist.second.cols() + j] = dist.second(i, j);
                }
            }
            
            tracks.push_back(tr);
        }
        return msgs;
    }

    void visualize(cv::Mat& frame, const std::vector<Observation::Ptr>& observations) 
    {
        cout << "----------" << endl;
        cv::putText(frame, (boost::format("Frames:%d") % messageNums).str(), cv::Point(20, 20), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 0), 4);
        for(const auto& observation: observations) {
            // cout << "Drawing Circle" << endl;
            // cout << "Odis: " << observation->bboxScaledWidth << endl;
            cv::rectangle(frame, cv::Rect(observation->x1, observation->y1, observation->width, observation->height), cv::Scalar(0,0,255), 2);
            cv::circle(frame, cv::Point(observation->centroid_u, observation->centroid_v), 5, cv::Scalar(0, 0, 255), -1);
        }

        for(const auto& person : people_tracker->get_people()) {
            if(!person->is_valid() || (person->last_associated==nullptr)) {
                continue;
            }
            auto dist = person->expected_measurement_distribution();
            // cout << "dist[0]" << endl << dist.first << endl;
            // cout << "dist[1]" << endl << dist.second << endl;
            cout << "ID  : " << (boost::format("%d") % person->id()).str() << endl;
            cout << "(" << person->last_associated->x1+5 <<" ," << person->last_associated->y1+15 << ")" << endl;
            cv::putText(frame, (boost::format("id:%d") % person->id()).str(), cv::Point(person->last_associated->x1+5, person->last_associated->y1+15), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0), 2);

            // cout << "Odis: " << (boost::format("%.2f") % person->last_associated->bboxScaledWidth).str() << endl;
            // cout << "width: " << (boost::format("%.2f") % person->last_associated->bboxWidth).str() << endl;
            cv::putText(frame, (boost::format("odis:%.2f") % person->last_associated->bboxScaledWidth).str(), cv::Point(person->last_associated->x1+5, person->last_associated->y1+30), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0), 2);
            
            // cout << "DIS : " << (boost::format("%.2f") % dist.first[1]).str() << endl;
            cv::putText(frame, (boost::format("dis:%.2f") % dist.first[1]).str(), cv::Point(person->last_associated->x1+5, person->last_associated->y1+45), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0), 2);

            // cout << "TF  : " << (boost::format("%.2f") % person->trace()).str() << endl;
            cv::putText(frame, (boost::format("tf:%.2f") % person->trace()).str(), cv::Point(person->last_associated->x1+5, person->last_associated->y1+60), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0), 2);
        }
        messageNums++;
        // Save image
        cout << "----------" << endl;

    }

    visualization_msgs::MarkerArrayConstPtr create_markers(const ros::Time& stamp) const
    {
        visualization_msgs::MarkerArrayPtr markers(new visualization_msgs::MarkerArray());
        if(!people_tracker)
            return markers;
        
        // cout << "Creating markers" << endl;
        const auto& people = people_tracker->get_people();
        markers->markers.resize(people.size() * 5);

        int i=0;
        for(const auto& person : people) {
            if(!person->is_valid()) {
                continue;
            }
            // cout << "Creating markers " << people[i]->id() << endl;
            const auto& color = color_palette[person->id() % color_palette.size()];
            std_msgs::ColorRGBA rgba;
            rgba.r = color[2] / 255.0f;
            rgba.g = color[1] / 255.0f;
            rgba.b = color[0] / 255.0f;
            rgba.a = 0.5f;
            // Body
            // cout << "Creating markers body " << people[i]->id() << endl;
            markers->markers[i].ns = "body";
            markers->markers[i].id = i;
            markers->markers[i].header.stamp = stamp;
            markers->markers[i].header.frame_id = "odom";
            markers->markers[i].action = visualization_msgs::Marker::ADD;
            markers->markers[i].lifetime = ros::Duration(1.0);
            markers->markers[i].type = visualization_msgs::Marker::CYLINDER;
            markers->markers[i].scale.x = 0.2f;
            markers->markers[i].scale.y = 0.2f;
            markers->markers[i].scale.z = 1.2f;
            markers->markers[i].pose.position.x = person->pos().x();
            markers->markers[i].pose.position.y = person->pos().y();
            markers->markers[i].pose.position.z = 0.8f;
            markers->markers[i].color = rgba;
            // head
            // cout << "Creating markers head " << people[i]->id() << endl;
            markers->markers[i+1] = markers->markers[i];
            markers->markers[i+1].ns = "head";
            markers->markers[i+1].id = i+1;
            markers->markers[i+1].type = visualization_msgs::Marker::SPHERE;
            markers->markers[i+1].scale.x = 0.2f;
            markers->markers[i+1].scale.y = 0.2f;
            markers->markers[i+1].scale.z = 0.2f;
            markers->markers[i+1].pose.position.z = 1.5f;
            // id
            // cout << "Creating markers id " << people[i]->id() << endl;
            markers->markers[i+2] = markers->markers[i+1];
            markers->markers[i+2].ns = "id";
            markers->markers[i+2].id = i+2;
            markers->markers[i+2].type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            markers->markers[i+2].text = to_string(person->id());
            markers->markers[i+2].scale.z = 0.2f;
            markers->markers[i+2].pose.position.z = 1.7f;
            markers->markers[i+2].color.r = 1.0f;
            markers->markers[i+2].color.g = 1.0f;
            markers->markers[i+2].color.b = 1.0f;
            markers->markers[i+2].color.a = 1.0f;
            // arrow pointing in direction they're facing with magnitude proportional to speed
            // cout << "Creating markers arrow " << people[i]->id() << endl;
            markers->markers[i+3] = markers->markers[i+2];
            markers->markers[i+3].ns = "direction";
            markers->markers[i+3].id = i+3;
            markers->markers[i+3].type = visualization_msgs::Marker::ARROW;
            Eigen::Vector2f pos = person->pos();
            Eigen::Vector2f vel = person->vel();
            geometry_msgs::Point startP, endP;
            startP.x = person->pos().x();
            startP.y = person->pos().y();
            endP.x = person->pos().x() + 0.5*person->vel().x();
            endP.y = person->pos().y() + 0.5*person->vel().y();
            markers->markers[i+3].points.push_back(startP);
            markers->markers[i+3].points.push_back(endP);
            markers->markers[i+3].scale.x = 0.05f;
            markers->markers[i+3].scale.y = 0.1f;
            markers->markers[i+3].scale.z = 0.2f;
            markers->markers[i+3].pose.position.x = 0.0f;
            markers->markers[i+3].pose.position.y = 0.0f;
            markers->markers[i+3].pose.position.z = 0.1f;
            markers->markers[i+3].color = rgba;

            i=i+4;
        }
        return markers;
    }
    
    ros::NodeHandle nh;
    ros::NodeHandle private_nh;

    std::shared_ptr<tf::TransformListener> tf_listener;

    ros::Subscriber poses_sub;
    ros::Subscriber camera_info_sub;

    ros::Publisher tracks_pub;
    ros::Publisher markers_pub;

    image_transport::ImageTransport image_trans;
    image_transport::Publisher image_pub;

    boost::circular_buffer<cv::Scalar> color_palette;

    sensor_msgs::CameraInfoConstPtr camera_info_msg;

    std::shared_ptr<TrackSystem> track_system;
    std::unique_ptr<PeopleTracker> people_tracker;

    int messageNums = 0;
};
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "mono_tracking");
  std::unique_ptr<mono_tracking::MonoTrackingNode> node(new mono_tracking::MonoTrackingNode());
  ros::spin();

  return 0;
}