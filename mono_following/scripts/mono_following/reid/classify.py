#! /usr/bin/env python3
import os
import copy
import rospy
import ros_numpy
import threading

from mono_following.srv import extractDescriptors, extractDescriptorsResponse
from mono_following.srv import classifyTarget, classifyTargetResponse

from mono_following.msg import Descriptor
from mono_following.msg import Samples

from yolox_descriptor.deep.targetClassifier import PersonClassifier
from yolox_descriptor.utils.meters import AverageMeter
from tuner import Tuner
_dir = os.path.split(os.path.realpath(__file__))[0]

class MonoClassifier(object):
    def __init__(self):
        self.classifier = PersonClassifier(model_path=os.path.join(_dir,"yolox_descriptor/deep/checkpoint/ckpt.t7"), use_reid=False)
        self.tuner = Tuner(copy.deepcopy(self.classifier.net), freeze=True)
        self.lock = threading.RLock()

        SRV_TOPIC = "/mono_following/classify_target"
        self.srv = rospy.Service(SRV_TOPIC, classifyTarget, self.predict)

        SUB_TOPIC = "/mono_following/training_samples"
        self.sub = rospy.Subscriber(SUB_TOPIC, Samples, self.update)

        rospy.loginfo("MonoClassifier is ready!")
        rospy.spin()



    def predict(self, req):
        rospy.loginfo("Classifying")
        images = []
        for img_msg in req.images:
            images.append(ros_numpy.numpify(img_msg))

        self.lock.acquire()
        confidences = self.classifier(images)
        self.lock.release()

        res = extractDescriptorsResponse()
        for confidence in confidences:
            res.confidences.append(confidence)
        return res
    
    def update(self, msg):
        pos_images = []
        for pos_img_msg in msg.pos_image_patches:
            pos_images.append(ros_numpy.numpify(pos_img_msg))
        neg_images = []
        for neg_img_msg in msg.neg_image_patches:
            neg_images.append(ros_numpy.numpify(neg_img_msg))
        print("pos image:", len(pos_images))
        print("neg image:", len(neg_images))
        if len(pos_images)>1:
            model_cache = copy.deepcopy(self.classifier.net)
            model_cache.reid = False
            model_cache = self.tuner.tune(model_cache, pos_images, neg_images)

            self.lock.acquire()
            self.classifier.net = copy.deepcopy(model_cache)
            self.lock.release()