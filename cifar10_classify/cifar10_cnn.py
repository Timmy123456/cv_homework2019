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
import matplotlib.pyplot as plt

# 定义对数据的预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('plane' , 'car' , 'bird' , 'cat' , 'deer' , 'dog' , 'frog' , 'horse' , 'ship' , 'truck')

# 用于画图的变量
epoch_value = list(0. for i in range(20))
loss_value = list(0. for i in range(20))
train_value = list(0. for i in range(20))
vali_value = list(0. for i in range(20))

# 定义cnn网络
class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # 利用view函数使得conv2层输出的16*5*5维的特征图尺寸变为400大小从而方便后面的全连接层的连接
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def train(model):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练模型
    epochs = 20
    for e in range(epochs):
        epoch_value[e] = e + 1
        train_loss = 0
        for data, target in trainloader:
            # 梯度清零
            optimizer.zero_grad()

            # forward+backward
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # 更新参数
            optimizer.step()

            train_loss += loss.item() * data.size(0)  # loss.item()是平均损失，平均损失*batch_size=一次训练的损失

        train_loss = train_loss / len(trainloader.dataset)

        logging.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
            ' Epoch: {} \t Training Loss:{:.6f}'.format(e + 1, train_loss))

        loss_value[e] = train_loss

        # 验证集的效果
        class_correct = list(0. for i in range(10))
        class_train_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        class_train_total = list(0. for i in range(10))
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

            for data in trainloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_train_correct[label] += c[i].item()
                    class_train_total[label] += 1

        correct = 0.0
        total = 0.0
        for i in range(10):
            logging.debug('\t Train accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_train_correct[i] / class_train_total[i]))
            correct += class_train_correct[i]
            total += class_train_total[i]

        train_value[e] = correct / total
        logging.debug('Train accuracy: %.6f' % train_value[e])

        correct = 0.0
        total = 0.0
        for i in range(10):
            logging.debug('\t Validation accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
            correct += class_correct[i]
            total += class_total[i]

        #global vali_value
        vali_value[e] = correct / total
        logging.debug('Validation accuracy: %.6f' % vali_value[e])



if __name__ == "__main__":
    # 打开log文件
    now = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
    name = "log_" + now + r".txt"
    logging.basicConfig(filename=os.path.join(os.getcwd(), name), level=logging.DEBUG)

    # 训练
    logging.debug("init model")
    model = classifier()
    logging.info(model)
    train(model)

    '''
    # 在服务器上运行时使用16线程
    num_processes = 1
    model = classifier()
    logging.info(model)

    model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    '''

    # 保存模型
    name = "cnn_" + now + r".pkl"
    torch.save(model.state_dict(), name)

    print(epoch_value)
    print(loss_value)
    print(train_value)
    print(vali_value)

    # 画图保存
    plt.title("loss_vs_epoch", fontsize=24)
    plt.xlabel("epoch", fontsize=10)
    plt.ylabel("loss", fontsize=10)
    plt.plot(epoch_value, loss_value, linewidth=5)
    name = "./loss_vs_epoch_" + now + r".png"
    plt.savefig(name)
    plt.clf()

    plt.title("accuracy_vs_epoch", fontsize=24)
    plt.xlabel("epoch", fontsize=10)
    plt.ylabel("accuracy", fontsize=10)
    plt.plot(epoch_value, train_value, linewidth=5)
    plt.plot(epoch_value, vali_value, linewidth=5)
    name = "./accuracy_vs_epoch_" + now + r".png"
    plt.savefig(name)
    plt.clf()
