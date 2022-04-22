#ifndef MONO_TRACKING_OBSERVATION_HPP
#define MONO_TRACKING_OBSERVATION_HPP


#include <memory>
#include <vector>
#include <math.h>

#include <ros/node_handle.h>
#include <opencv2/core/core.hpp>
#include <boost/optional.hpp>
#include <sensor_msgs/CameraInfo.h>

#include <mono_tracking/track_system.hpp>

using namespace std;
namespace mono_tracking
{
struct Observation
{
public:
    using Ptr = std::shared_ptr<Observation>;
    Observation(ros::NodeHandle& private_nh, const std::shared_ptr<TrackSystem>& _track_system, const sensor_msgs::CameraInfoConstPtr& camera_info_msg, const vector<u_int16_t> &box_xywh, float _score)
    : k(_track_system->k), b(_track_system->b), score(_score)
    {
        int border_thresh_w = private_nh.param<int>("detection_border_thresh_w", 20);
        init_box(box_xywh, camera_info_msg->width, camera_info_msg->height);
        // int border_thresh_h = private_nh.param<int>("detection_border_thresh_h", 25);
        // Our observation
        // bboxCentroid_u = x1+width/2;
        // bboxCentroid_v = y1+height/2;
        // bboxScaledWidth = (bboxWidth-b)/k;  // linear regression model

        // bboxScaledWidth = exp(k*width+b);  // exponential regression model

        // transform
        Eigen::Matrix3f K;
        Eigen::Matrix4f Tcr, Trw;
        K = _track_system->camera_matrix;
        // Tcr = _track_system->footprint2camera.matrix();
        // Trw = _track_system->odom2footprint.matrix();
        Tcr = _track_system->base2camera.matrix();
        Trw = _track_system->odom2base.matrix();

        float real_width = private_nh.param<float>("real_width", 0.55);  // last 0.45
        bboxScaledWidth = K(0,0)*real_width/float(width);


        // The box's width can't be closed to the border
        // Because the boxWidth-distance will be inaccurate
        // But for person identification evaluation, this should be opened up

        // if(x1 > border_thresh_w && x2 < (camera_info_msg->width - border_thresh_w))
        // {
        //     // cout << "I'm good: " << x2 << " " << camera_info_msg->width - border_thresh_w << endl;
        //     isGoodObservation = true;
        //     // Our observation
        //     // descriptor = convertVector2Mat(rawDescriptor, 1, rawDescriptor.size());
        //     obs(0) = bboxScaledWidth*(centroid_u-K(0,2))/K(0,0)-Tcr(0,3)-Eigen::Vector3f(1.0f, 0.0f, 0.0f).transpose()*Tcr.block<3,3>(0,0)*Trw.block<3,1>(0,3);
        //     obs(1) = bboxScaledWidth-Tcr(2,3)-Eigen::Vector3f(0.0f, 0.0f, 1.0f).transpose()*Tcr.block<3,3>(0,0)*Trw.block<3,1>(0,3);
        // }

        isGoodObservation = true;
        // Our observation
        obs(0) = bboxScaledWidth*(centroid_u-K(0,2))/K(0,0)-Tcr(0,3)-Eigen::Vector3f(1.0f, 0.0f, 0.0f).transpose()*Tcr.block<3,3>(0,0)*Trw.block<3,1>(0,3);
        obs(1) = bboxScaledWidth-Tcr(2,3)-Eigen::Vector3f(0.0f, 0.0f, 1.0f).transpose()*Tcr.block<3,3>(0,0)*Trw.block<3,1>(0,3);
    }

    void init_box(const vector<u_int16_t> &xywh, int img_width, int img_height){
        centroid_u = xywh[0];
        centroid_v = xywh[1];
        width = xywh[2];
        height = xywh[3];
        x1 = max(int(centroid_u-width/2),0);
        x2 = min(int(centroid_u+width/2),img_width-1);
        y1 = max(int(centroid_v-height/2),0);
        y2 = min(int(centroid_v+height/2),img_height-1);
    }

    // cv::Mat convertVector2Mat(vector<float> v, int channels, int rows)
    // {
    //     cv::Mat mat = cv::Mat(v);//将vector变成单列的mat
    //     cv::Mat dest = mat.reshape(channels, rows).clone();//PS：必须clone()一份，否则返回出错
    //     return dest;
    // }

    // params
    double k,b;

    // raw observation
    int centroid_u,centroid_v,width,height;
    int x1,y1,x2,y2;
    float score;

    // processed observation
    double bboxScaledWidth;
    Eigen::Vector2f obs;

    bool isGoodObservation=false;

};
}
#endif // OBSERVATION_HPP