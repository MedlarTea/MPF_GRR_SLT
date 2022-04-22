#! /usr/bin/env python3

import os
import rospy
import ros_numpy
import cv2
import time
# from sensor_msgs.msg import 
from mono_tracking.msg import Box
from mono_tracking.msg import BoxArray

class Initializer:
    def __init__(self):
        BOXES_TOPIC = "/detect/image_boxes"
        self.boxesSub = rospy.Subscriber(BOXES_TOPIC, BoxArray, self.boxesCallback)

