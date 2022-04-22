from geometry_msgs.msg import PointStamped
import cv2
import numpy as np

class Tracklet:
    def __init__(self, header, track_msg, image):
        ### prior parameters ### 
        self.down_width = 64
        self.down_height = 128

        ### information ###
        self.region = None
        self.image_patch = None
        self.descriptor = None
        self.target_confidence = None

        ### transform the point ###
        person_point = PointStamped()
        person_point.header = header
        person_point.point = track_msg.pos
        # transformed_point = PointStamped()
        # tf_listener.transformPoint("base_link", person_point, transformed_point)
        # self.pos_in_baselink = [transformed_point.point.x, transformed_point.point.y]
        self.pos_in_baselink = [person_point.point.x, person_point.point.y]
        self.distance = np.linalg.norm(self.pos_in_baselink)

        # print(track_msg.box.box)
        if len(track_msg.box.box)!=0:
            # u_tl, v_tl, u_br, v_br
            self.region = track_msg.box.box
            self.image_patch = self.preprocessImage(image, self.region)
            # print(self.image_patch.shape)

    def preprocessImage(self, image, patch):
        image = image[patch[1]:patch[3], patch[0]:patch[2], :]
        image = cv2.resize(image, (self.down_width, self.down_height), interpolation=cv2.INTER_LINEAR)
        return image
    


