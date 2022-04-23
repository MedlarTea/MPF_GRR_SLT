#! /usr/bin/env python3

import os
import rospy
import ros_numpy
import cv2
import time
import numpy as np
import shutil
# from sensor_msgs.msg import 
# from geometry_msgs import Point
# from geometry_msgs import Vector3
from mono_tracking.msg import Track
from mono_tracking.msg import TrackArray
from tensorboardX import SummaryWriter

_dir = os.path.split(os.path.realpath(__file__))[0]

class Monitor:
    def __init__(self):
        TRACKS_TOPIC = "/mono_tracking/tracks"
        self.boxesSub = rospy.Subscriber(TRACKS_TOPIC, TrackArray, self.tracksCallback)

        path = os.path.join(_dir, 'logs')
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            shutil.rmtree(path)
        self.writer = SummaryWriter(path)
        self.nums = 0

        # To be monitored
        self.positions = []
        self.velocities = []
        self.traces = []
        self.covs = []
        self.last_associateds = []

        rospy.loginfo("Monitor is ready!")
        rospy.spin()
        
    

    def tracksCallback(self, trackArrayMsg):
        for track in trackArrayMsg.tracks:
            if track.id ==1:
                continue
            # Read
            # self.positions.append([track.pos.x, track.pos.y])
            # self.velocities.append([track.vel.x, track.vel.y])
            # self.traces.append(track.trace)
            # self.covs.append(np.array(track.cov).reshape(4,4))
            cov = np.array(track.cov).reshape(4,4)
            # self.last_associateds.append(track.last_associated)

            # Monitor
            self.writer.add_scalars("Target position", {'X': track.pos.x}, self.nums)
            self.writer.add_scalars("Target position", {'Y': track.pos.y}, self.nums)
            self.writer.add_scalars("Target velocity", {'X': track.vel.x}, self.nums)
            self.writer.add_scalars("Target velocity", {'Y': track.vel.y}, self.nums)
            self.writer.add_scalars("Target observation", {'obs1': track.last_associated[0]}, self.nums)
            self.writer.add_scalars("Target observation", {'obs2': track.last_associated[1]}, self.nums)
            self.writer.add_scalars("Target trace", {'trace': track.trace}, self.nums)
            self.writer.add_scalars("Target cov", {'(0,0)': cov[0][0]}, self.nums)
            self.writer.add_scalars("Target cov", {'(1,1)': cov[1][1]}, self.nums)
            self.writer.add_scalars("Target cov", {'(2,2)': cov[2][2]}, self.nums)
            self.writer.add_scalars("Target cov", {'(3,3)': cov[3][3]}, self.nums)


            self.nums+=1


if __name__ == '__main__':
    rospy.init_node('mono_monitor', anonymous=True)
    monoDetector = Monitor()



