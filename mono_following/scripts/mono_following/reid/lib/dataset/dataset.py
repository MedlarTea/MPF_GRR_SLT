import numpy as np
from PIL import Image
import os.path as osp
import json
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

class MyDataset(Dataset):
    def __init__(self):
        self.norm = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.trainSet = []

    def load(self, posTrainSet, negTrainSet):
        self.trainSet = []
        for pos in posTrainSet:
            self.trainSet.append((pos, 0))
        for neg in negTrainSet:
            self.trainSet.append((neg, 1))
    
    def __len__(self):
        return len(self.trainSet)
    
    def __getitem__(self,indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        img, label = self.trainSet[index]

        img = self._preprocess(img)
        return img, label

    def _preprocess(self, im):
        return self.norm(im.astype(np.float32)/255.)
        
        



