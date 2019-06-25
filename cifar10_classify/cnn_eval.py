# -*- coding:utf-8 -*-

import numpy as np
import os
import torch
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.multiprocessing as mp
import time
import logging
import cifar10_cnn

testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=cifar10_cnn.transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 加载模型
model = cifar10_cnn.classifier()
model.load_state_dict(torch.load('cnn_model.pkl'))

# 验证集的效果
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))