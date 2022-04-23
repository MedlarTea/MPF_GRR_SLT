#! /usr/bin/env python3

import os
import rospy
import ros_numpy
import cv2
import time
from sensor_msgs.msg import Image
from mono_tracking.msg import Box
from mono_tracking.msg import BoxArray

from yolox.YOLOX.detector import PersonDetector
from yolox.utils.meters import AverageMeter

_dir = os.path.split(os.path.realpath(__file__))[0]


class MonoDetector:
    def __init__(self):
        self.detector = PersonDetector(model='yolox-m', ckpt=os.path.join(_dir, 'yolox_descriptor/weights/yolox_m.pth.tar'))
        # self.extractor = PersonExtractor(model_path=os.path.join(_dir,"yolox_descriptor/deep/checkpoint/ckpt.t7"))

        IMAGE_TOPIC = "/camera/color/image_raw"
        self.imageSub = rospy.Subscriber(IMAGE_TOPIC, Image, self.imageCallback)

        BOXES_TOPIC = "people/boxes"
        self.boxesPub = rospy.Publisher(BOXES_TOPIC, BoxArray, queue_size=1)

        VISUALIZE_TOPIC = "/detect/image_boxes"
        self.imageBoxesPub = rospy.Publisher(VISUALIZE_TOPIC, Image, queue_size=1)

        self.detect_time = AverageMeter()
        self.extract_time = AverageMeter()
        self.once_time = AverageMeter()
        self.nums = 0
        self.print_freq = 100
        rospy.loginfo("MonoDetector is ready!")
        rospy.spin()
    
    def imageCallback(self, imgMsg):
        """
        In my RTX2060, time-0.05s(20Hz), detect-0.039/0.050, extract-0.009/0.050
        """
        once_end = time.time()
        image = ros_numpy.numpify(imgMsg)
        cv2.imwrite("./test.jpg", image)

        # detect
        detect_end = time.time()
        xywh, scores = self.detector.detect(image, conf=0.7)
        self.detect_time.update(time.time()-detect_end)
        # extract
        # extract_end = time.time()
        # xyxy, features = self.extractor(xywh, image)
        # self.extract_time.update(time.time()-extract_end)

        boxes = []
        boxArray = BoxArray()
        for (x1,y1,w,h), score in zip(xywh, scores):
        # for _box, score in zip(xywh, scores):
            box = Box()
            box.box = [int(x1),int(y1),int(w),int(h)]
            box.score = score
            # box.descriptor = feature
            boxes.append(box)
        boxArray.boxes = boxes
        boxArray.header.frame_id = imgMsg.header.frame_id
        # boxArray.header.stamp = imgMsg.header.stamp = rospy.Time.now()  # record the latest time
        boxArray.header.stamp = imgMsg.header.stamp  # record original time, for distance estimation 
        boxArray.image = imgMsg
        # boxArray.image = self.visualize(image, boxes)
        self.boxesPub.publish(boxArray)
        
        self.once_time.update(time.time()-once_end)
        if((self.nums+1)%self.print_freq ==0):
            rospy.loginfo('DeTime {:.3f} ({:.3f})\t'
                          'OnTime {:.3f} ({:.3f})\t'.format(
                           self.detect_time.val, self.detect_time.avg,
                           self.once_time.val, self.once_time.avg
                        ))
        self.nums+=1
    
    def xywh_to_xyxy(self, bbox_xywh, img_width, img_height):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),img_width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),img_height-1)
        return x1,y1,x2,y2
    
    def visualize(self, image, boxes):
        visualized_image = image
        for box in boxes:
            x1,y1,x2,y2 = self.xywh_to_xyxy(box.box, image.shape[1], image.shape[0])
            visualized_image = cv2.rectangle(visualized_image,(x1,y1), (x2,y2), (255, 0, 0), 2)
        imgMsg = ros_numpy.msgify(Image, visualized_image, encoding="rgb8")
        return imgMsg
        # self.imageBoxesPub.publish(imgMsg)


if __name__ == '__main__':
    rospy.init_node('mono_detector', anonymous=True)
    monoDetector = MonoDetector()

