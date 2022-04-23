#! /usr/bin/env python3
import numpy as np
import ros_numpy
import cv2
from tqdm import main
import rospy

# My class
from tracklet import Tracklet
from states.initial_state import InitialState
from descriminator import Descriminator
# standard ROS message
from sensor_msgs.msg import Image
# import tf

# my ROS message
from mono_tracking.msg import TrackArray
from mono_tracking.msg import Track
from mono_following.msg import Target
# from mono_following.msg import Box


class MonoFollowing:
    def __init__(self):
        ### Some parameters for debug and testing ###
        self.SAVE_TRACKS = False
        self.SAVE_IMAGES = False
        self.IMAGE_PATH = "/home/jing/Data/Dataset/FOLLOWING/ccf_failure_dataset_masaic/ccf_failure_dataset/room433/debug/grr_slt"
        self.STORE_DIR = rospy.get_param("~track_store_dir")

        ### Our parameters ###
        self.state = InitialState()
        self.descriminator = Descriminator()

        ### Our necessary topics ###
        # listen to the transform tree
        # self.tf_listener = tf.TransformListener()

        # subscribe tracks from mono_tracking
        self.tracks_sub = rospy.Subscriber("/mono_tracking/tracks", TrackArray, self.callback)
        # publish image for visualization
        self.image_pub = rospy.Publisher("/mono_following/vis_image", Image, queue_size=1)
        # publish image patches for visualization
        self.patches_pub = rospy.Publisher("/mono_following/patches", Image, queue_size=1)
        # publish target information
        self.target_pub = rospy.Publisher("/mono_following/target", Target, queue_size=1)

        ### Node is already set up ###
        rospy.loginfo("Mono Following Node is Ready!")
        rospy.spin()
    
    def callback(self, tracks_msg):
        # print("CALLBACK")
        # get messages
        self.tracks = {}
        if tracks_msg.image.encoding == "rgb8":
            image = ros_numpy.numpify(tracks_msg.image)
        elif tracks_msg.image.encoding == "bgr8":
            image = ros_numpy.numpify(tracks_msg.image)[:,:,[2,1,0]]  # change to rgb

        for track in tracks_msg.tracks:
            self.tracks[track.id] = Tracklet(tracks_msg.header, track, image)

        # extract features
        self.descriminator.extractFeatures(self.tracks)
        
        # update the state
        # print(self.state.state_name())
        next_state = self.state.update(self.descriminator, self.tracks)
        if next_state is not self.state:
            self.state = next_state
        
        # publish target information
        if self.target_pub.get_num_connections():
            target = Target()
            target.header = tracks_msg.header
            target.state.data = self.state.state_name()
            target.target_id = self.state.target()
            
            track_ids = []
            confidences = []
            for idx in self.tracks.keys():
                track_ids.append(idx)
                if idx == target.target_id and self.tracks[idx].target_confidence != None:
                    target.box = self.tracks[idx].region # u_tl, v_tl, u_br, v_br
                    target.classifier_confidences = [self.tracks[idx].target_confidence]
                    target.position = self.tracks[idx].pos_in_baselink
                if self.tracks[idx].target_confidence != None:
                    confidences.append(self.tracks[idx].target_confidence)
            target.track_ids = track_ids
            target.confidences = confidences
            self.target_pub.publish(target)
        
        # publish image containing the identification result
        if self.image_pub.get_num_connections():
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_bgr = self.visualize(image_bgr, self.tracks)
            image_msg = ros_numpy.msgify(Image, image_bgr, encoding = "bgr8")
            self.image_pub.publish(image_msg)
            if self.SAVE_IMAGES:
                pass
        # publish target patch to see whether is OK
        if self.patches_pub.get_num_connections():
            image = self.visualize_patches(self.tracks)
            if image is not None:
                # cv2.imwrite("./test.jpg", image)
                image_msg = ros_numpy.msgify(Image, image, encoding = "rgb8")
                self.patches_pub.publish(image_msg)

        # Save tracks information, for debug and analysis
        if self.SAVE_TRACKS:
            pass

    def visualize(self, original_image, tracks):
        # print("VISUALIZE")
        image = original_image.copy()
        for idx in tracks.keys():
            if tracks[idx].target_confidence == None:
                continue
            image = cv2.putText(original_image, self.state.state_name(), (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            if idx == self.state.target():
                image = cv2.rectangle(image, (self.tracks[idx].region[0],self.tracks[idx].region[1]), (self.tracks[idx].region[2],self.tracks[idx].region[3]), (0,255,0), 2)
                image = cv2.putText(image, "id:{:d}".format(idx), (int((self.tracks[idx].region[0]+self.tracks[idx].region[2])/2-5), int((self.tracks[idx].region[1]+self.tracks[idx].region[3])/2-5)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
            else:
                image = cv2.rectangle(image, (self.tracks[idx].region[0],self.tracks[idx].region[1]), (self.tracks[idx].region[2],self.tracks[idx].region[3]), (0,0,255), 2)
                image = cv2.putText(image, "id:{:d}".format(idx), ((int((self.tracks[idx].region[0]+self.tracks[idx].region[2])/2-5), int((self.tracks[idx].region[1]+self.tracks[idx].region[3])/2-5))), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
        return image
    
    def saveTrack(self, track_id, track, store_path):
        print("SAVE TRACKS")

    def visualize_patches(self, tracks):
        target_id = self.state.target()
        if target_id in tracks.keys() and self.tracks[target_id].image_patch is not None:
            return self.tracks[target_id].image_patch
        else:
            return None
if __name__ == "__main__":
    rospy.init_node('mono_following', anonymous=True)
    mono_following = MonoFollowing()

