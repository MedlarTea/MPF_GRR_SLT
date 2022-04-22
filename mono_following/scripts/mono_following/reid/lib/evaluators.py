import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from .utils.meters import AverageMeter
import numpy as np


class ClassificationEvaluator(object):
    def __init__(self, model, gpu):
        super(ClassificationEvaluator, self).__init__()
        self.model = model
        self.gpu = gpu
    
    def evaluate(self, data_loader):
        self.model.eval()

        acces = AverageMeter()
        wrong_predictions = np.array([])
        for i, inputs in enumerate(data_loader):
            inputs, labels, fnames = self._parse_data(inputs)
            acc, correct = self._forward(inputs, labels)
            acces.update(acc)
            wrong_predictions = np.append(wrong_predictions, np.array(fnames)[correct==0])
        print('ACC {:.3f}'.format(acces.avg))
        # print(wrong_predictions)
    
    def _parse_data(self, inputs):
        imgs = inputs[0]
        labels = inputs[1]
        fnames = inputs[2]
        return imgs.cuda(self.gpu), labels.cuda(self.gpu), fnames

    def _forward(self, inputs, labels):
        preds = self.model(inputs)
        acc, correct = self.calculate_accuracy(preds, labels)
        return acc, correct
    
    def calculate_accuracy(self, y_pred, y):
        top_pred = y_pred.argmax(1, keepdim = True)
        correct = top_pred.eq(y.view_as(top_pred))
        acc = correct.sum()
        acc = acc.float() / y.shape[0]
        return acc, np.array(correct.squeeze(1).to(dtype=torch.int, device="cpu"))

    

    