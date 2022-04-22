import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from .utils.meters import AverageMeter

class ClassificationTrainer(object):
    def __init__(self, model, gpu=None):
        super(ClassificationTrainer, self).__init__()
        self.model = model
        self.gpu = gpu
        self.criterion = nn.CrossEntropyLoss()  # 交叉熵
        self.softmax = nn.Softmax(dim=1)
    
    def train(self, epoch, data_loader, optimizer, train_iters, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acces = AverageMeter()
        end = time.time()

        for i, inputs in enumerate(data_loader):

            inputs, labels = self._parse_data(inputs)
            data_time.update(time.time() - end)

            loss, acc = self._forward(inputs, labels)
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            acces.update(acc)

            batch_time.update(time.time() - end)
            end = time.time()

            try:
                rank = dist.get_rank()
            except:
                rank = 0
            if ((i + 1) % print_freq == 0 and rank==0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'ACC {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              acces.val, acces.avg))
        return acces.avg
    
    def _parse_data(self, inputs):
        imgs = inputs[0]
        labels = inputs[1]
        return imgs.cuda(self.gpu), labels.cuda(self.gpu)

    def _forward(self, inputs, labels):
        preds = self.model(inputs)
        # print(preds)
        # print(labels)
        loss = self.criterion(preds, labels)
        preds = self.softmax(preds)
        acc = self.calculate_accuracy(preds, labels)
        return loss, acc
    
    def calculate_accuracy(self, y_pred, y):
        top_pred = y_pred.argmax(1, keepdim = True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc

    