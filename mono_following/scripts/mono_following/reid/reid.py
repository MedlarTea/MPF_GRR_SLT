#! /usr/bin/env python3
import os
import copy
import rospy
import ros_numpy
import time
import threading
import numpy as np
from sklearn.linear_model import Ridge
from scipy.spatial import distance


from mono_following.srv import updateDescriptors, updateDescriptorsResponse
from mono_following.srv import classifyTarget, classifyTargetResponse
# from mono_following.msg import Descriptor

from yolox_descriptor.deep.feature_extractor import PersonExtractor
from yolox_descriptor.utils.meters import AverageMeter
from tuner import Tuner
from RRClassifier import RRClassifier, RRClassifierWithStrategy
_dir = os.path.split(os.path.realpath(__file__))[0]

np.random.seed(3)

class MonoReid:
    def __init__(self):
        self.extractor = PersonExtractor(model_path=os.path.join(_dir,"yolox_descriptor/deep/checkpoint/ckpt.t7"))
        # self.tuner = Tuner(copy.deepcopy(self.extractor.net))
        self.lock = threading.RLock()

        # update
        UPDATE_SRV = "/mono_following/update_descriptors"
        self.update_srv = rospy.Service(UPDATE_SRV, updateDescriptors, self.update)
        alpha = 1.0
        samples = 64
        threshold = 0.02
        # for short-term classification model
        # self.clf_st = RRClassifier(alpha, samples)
        # for long-term classification model
        self.clf_lt = RRClassifierWithStrategy(alpha, samples, threshold)
        # for model choosing
        self.clf = None
        self.st_nums = self.lt_nums = 0

        # classify
        CLASSIFY_SRV = "/mono_following/classify_target"
        self.classify_srv = rospy.Service(CLASSIFY_SRV, classifyTarget, self.classify_target)

        # record time
        self.extract_time = AverageMeter()
        self.cache_time = AverageMeter()
        self.classifier_time = AverageMeter()
        self.once_time = AverageMeter()
        self.nums = 0
        self.print_freq = 100

        rospy.loginfo("MonoReid is ready!")
        rospy.spin()

    def update(self, req):
        once_end = time.time()
        # rospy.loginfo("Updating Features")
        images = []
        boxes = []
        distances = []
        labels = []
        for i in range(len(req.images)):
            images.append(ros_numpy.numpify(req.images[i]))
            labels.append(req.labels[i])
            boxes.append(req.boxes[i].box)
            distances.append(req.distances[i])


        self.lock.acquire()

        extract_end = time.time()
        descriptors = self.extractor(images)  # extract features
        features = []
        for i in range(len(descriptors)):
            features.append([descriptors[i], np.array(distances[i]), np.asarray([boxes[i]])])
        self.extract_time.update(time.time() - extract_end)

        cache_end = time.time()
        # self.clf_st.update_cache(features, labels)  # update st-cache
        output = self.clf_lt.update_cache(features, labels)  # update lt-cache
        self.cache_time.update(time.time() - cache_end)

        classifier_end = time.time()
        # self.clf_st.update_classifier()  # update short-term classifier
        self.clf_lt.update_classifier()  # update long_term classifier
        self.classifier_time.update(time.time() - classifier_end)

        # choose model
        # score_lt = score_st = 0.0
        # score_pos_lt = score_neg_lt = score_pos_st = score_neg_st = 0.0
        # pos = neg = 0
        # if self.clf == None:
        #     self.clf = self.clf_st
        # for i in range(len(labels)):
        #     if labels[i] == 1:
        #         score_pos_lt += self.clf_lt.predict(descriptors[i])
        #         score_pos_st += self.clf_st.predict(descriptors[i])
        #         pos+=1
        #     elif labels[i] == 0:
        #         score_neg_lt += self.clf_lt.predict(descriptors[i])
        #         score_neg_st += self.clf_st.predict(descriptors[i])
        #         neg+=1
        # if pos>0:
        #     score_pos_lt = score_pos_lt/pos
        #     score_pos_st = score_pos_st/pos
        # if neg>0:
        #     score_neg_lt = score_neg_lt/neg
        #     score_neg_st = score_neg_st/neg
        # score_lt = score_pos_lt - score_neg_lt
        # score_st = score_pos_st - score_neg_st
        # if score_lt > score_st:
        #     self.lt_nums+=1
        # else:
        #     self.st_nums+=1
        # if self.lt_nums == 5:
        #     self.clf = self.clf_lt
        #     self.lt_nums = self.st_nums = 0
        # if self.st_nums == 5:
        #     self.clf = self.clf_st
        #     self.lt_nums = self.st_nums = 0

        self.lock.release()

        res = updateDescriptorsResponse()
        res.isSuccessed = True

        self.once_time.update(time.time()-once_end)
        if((self.nums+1)%self.print_freq ==0):
            rospy.loginfo('ExTime {:.3f} ({:.3f})\t'
                          'CaTime {:.3f} ({:.3f})\t'
                          'UpTime {:.3f} ({:.3f})\t'
                          'OnTime {:.3f} ({:.3f})\t'.format(
                           self.extract_time.val, self.extract_time.avg,
                           self.cache_time.val, self.cache_time.avg,
                           self.classifier_time.val, self.classifier_time.avg,
                           self.once_time.val, self.once_time.avg
                        ))
        self.nums+=1
        return res

    def classify_target(self, req):
        # rospy.loginfo("Classifying target")
        images = []
        for i in range(len(req.images)):
            images.append(ros_numpy.numpify(req.images[i]))

        confidences = []
        self.lock.acquire()
        features = self.extractor(images)
        for feature in features:
            # score_st = self.clf_st.predict(feature)
            score_lt = self.clf_lt.predict(feature)
            score = score_lt
            # score = self.clf.predict(feature)
            confidences.append(score)
        self.lock.release()

        res = classifyTargetResponse()
        res.confidences = confidences
        return res
        
if __name__ == '__main__':
    rospy.init_node('MonoReid', anonymous=True)
    monoDetector = MonoReid()
    
    

