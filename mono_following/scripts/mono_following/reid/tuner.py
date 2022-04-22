import numpy as np
from PIL import Image
import os.path as osp
import copy
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from lib.trainers import ClassificationTrainer
from lib.dataset.dataset import MyDataset
class Tuner:
    def __init__(self, model, freeze=False):
        self.model = model
        self.trainer = ClassificationTrainer(model)
        self.dataset = MyDataset()
        self.bs = 4
        self.lr = 0.0001
        self.epochs = 50
        self.freeze = freeze
        self.print_freq = 1
        if self.freeze:
            self.freeze_model()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = self.lr)
        self.setup_seed(123)
    
    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def freeze_model(self):
        for i, p in enumerate(self.model.named_parameters()):
            # p is (module_name, param)
            if p[0].split(".")[0] in ["conv", "layer1", "layer2", "layer3", "layer4"]:
                p[1].requires_grad = False
    
    def get_data(self, posTrainSet, negTrainSet):
        self.dataset.load(posTrainSet, negTrainSet)
        return DataLoader(self.dataset, shuffle=True, batch_size=self.bs)

    def tune(self, model, posTrainSet, negTrainSet):
        self.model = model
        if self.freeze:
            self.freeze_model()
        train_loader = self.get_data(posTrainSet, negTrainSet)
        self.trainer = ClassificationTrainer(self.model)

        for epoch in range(0, self.epochs):
            acc = self.trainer.train(epoch, train_loader, self.optimizer, len(train_loader), self.print_freq)
        
        self.model.eval() 
        return self.model


