#! /usr/bin/env python3
import os
import copy
import rospy
import ros_numpy
import threading
from sklearn.linear_model import Ridge

from mono_following.srv import extractDescriptors, extractDescriptorsResponse
from mono_following.srv import classifyTarget, classifyTargetResponse

from mono_following.msg import Descriptor
from mono_following.msg import Samples

from yolox_descriptor.deep.feature_extractor import PersonExtractor
from yolox_descriptor.utils.meters import AverageMeter
from tuner import Tuner
_dir = os.path.split(os.path.realpath(__file__))[0]

class MonoExtractor:
    def __init__(self):
        self.extractor = PersonExtractor(model_path=os.path.join(_dir,"yolox_descriptor/deep/checkpoint/ckpt.t7"))
        self.tuner = Tuner(copy.deepcopy(self.extractor.net))
        self.lock = threading.RLock()

        SRV_TOPIC = "/mono_following/extract_descriptors"
        self.srv = rospy.Service(SRV_TOPIC, extractDescriptors, self.extract)

        # SUB_TOPIC = "/mono_following/training_samples"
        # self.sub = rospy.Subscriber(SUB_TOPIC, Samples, self.update)

        # Ridge regression model for online-learning-classification model
        self.clf = Ridge(alpha=0.0)

        rospy.loginfo("MonoExtractor is ready!")
        rospy.spin()

    def extract(self, req):
        rospy.loginfo("Extracting Features")
        images = []
        for img_msg in req.images:
            images.append(ros_numpy.numpify(img_msg))

        self.lock.acquire()
        features = self.extractor(images)
        self.lock.release()

        res = extractDescriptorsResponse()
        for feature in features:
            desc = Descriptor(feature)
            res.descriptors.append(desc)
        return res
    
    # def update(self, msg):
    #     pos_images = []
    #     for pos_img_msg in msg.pos_image_patches:
    #         pos_images.append(ros_numpy.numpify(pos_img_msg))
    #     neg_images = []
    #     for neg_img_msg in msg.neg_image_patches:
    #         neg_images.append(ros_numpy.numpify(neg_img_msg))
    #     print("pos image:", len(pos_images))
    #     print("neg image:", len(neg_images))
    #     if len(pos_images)>1:
    #         model_cache = copy.deepcopy(self.extractor.net)
    #         model_cache.reid = False
    #         model_cache = self.tuner.tune(model_cache, pos_images, neg_images)

    #         self.lock.acquire()
    #         self.extractor.net = copy.deepcopy(model_cache)
    #         self.extractor.net.reid = True
    #         self.lock.release()        
        
if __name__ == '__main__':
    rospy.init_node('mono_extractor', anonymous=True)
    monoDetector = MonoExtractor()
    
    

