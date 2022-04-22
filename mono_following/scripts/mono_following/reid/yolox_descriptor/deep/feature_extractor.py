import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .model import Net

import torch.nn as nn
class Extractor(object):
    def __init__(self, model_path, use_reid=True, use_cuda=True):
        self.net = Net(reid=use_reid)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        self.net.classifier[-1] = nn.Linear(256, 2)

        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        


    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """

        im_batch = torch.cat([self.norm(im.astype(np.float32)/255.).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


class PersonExtractor(Extractor):
    def __init__(self, model_path, use_reid=True, use_cuda=True):
        Extractor.__init__(self, model_path=model_path, use_reid=use_reid, use_cuda=use_cuda)
    
    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _get_feature(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()

    def __call__(self, imgs):
        if imgs:
            features = self._get_feature(imgs)
        else:
            features = np.array([])
        return features



if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

