# ============================================================================
#SF learning 

# simle split

# dataset: minist
# model: cnn

# ============================================================================

import torch
from torch import nn
import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from pandas import DataFrame

import random
import numpy as np
import os

import http.server
import socketserver
import threading
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
ID = 0

#===================================================================
# program = "SFLV1 Minist on CNN for client 0"
# print(f"---------{program}----------")   

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))     



#===================================================================
# parameter setting
#===================================================================
# No. of users
NUM_USERS = 5
EPOCH = 5
FRAC = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
LR = 0.0001
BATCH_SIZE = 50 # 批量训练时候一次送入数据的size

#===================================================================
# client side model difination
#===================================================================

class CNN_client(nn.Module):
    
    def __init__(self):
        pass
        super(CNN_client, self).__init__()

        self.conv1 =nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        out = x

        return out
    
#===================================================================
# dataset process
#===================================================================

DOWNLOAD_MNIST = True 
 
# 下载mnist手写数据集
# 训练集
train_data = torchvision.datasets.MNIST(  
    root = './MNIST/',                      
    train = True,                            
    transform = torchvision.transforms.ToTensor(),                                                
    download=DOWNLOAD_MNIST 
)
 
# 测试集
test_data = torchvision.datasets.MNIST(root='./MNIST/', train=False)  # train设置为False表示获取测试集
 
# 一个批训练 50个样本, 1 channel通道, 图片尺寸 28x28 size:(50, 1, 28, 28)
train_loader = DataLoader(
    dataset = train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
) 
#  测试数据预处理；只测试前2000个
test_x = torch.unsqueeze(test_data.data,dim=1).float()[:2000] / 255.0
# shape from (2000, 28, 28) to (2000, 1, 28, 28)
test_y = test_data.targets[:2000]

train_x = []
train_y = []

for _ in range(NUM_USERS):
    train_x_split = []
    train_x.append(train_x_split)

for _ in range(NUM_USERS):
    train_y_split = []
    train_y.append(train_x_split)
 
for step,(batch_x,batch_y) in enumerate(train_loader):

    train_x[step % NUM_USERS].append(batch_x)
    train_y[step % NUM_USERS].append(batch_y)


# my_train_x = train_x[ID]
# my_train_y = train_y[ID]


#===================================================================
# Data separete method
#===================================================================

# class DatasetSplit(Dataset):
#     #DatasetSplit 类继承自 Dataset 类，它的作用是将数据集按照给定的索引进行划分。
#     def __init__(self, dataset, idxs):
#         #__init__ 方法用于初始化类的实例，它接受两个参数：数据集和索引列表。
#         self.dataset = dataset
#         self.idxs = list(idxs)

#     def __len__(self):
#         #__len__ 方法返回划分后数据集的大小
#         return len(self.idxs)

#     def __getitem__(self, item):
#         #__getitem__ 方法根据给定的索引返回数据集中对应的图像和标签
#         image, label = self.dataset[self.idxs[item]]
#         return image, label



#===================================================================
# Client model
#===================================================================

class Client(object):
    def __init__(self, id , train_x, train_y, test_x, test_y):
        self.id =id
        self.train_x = train_x[id]
        self.train_y = train_y[id]
        self.test_x = test_x
        self.test_y = test_y

        self.net = CNN_client()
    
    # def train(self, net):
    #     optimizer_1 = torch.optim.Adam(net.parameters(),lr=LR) 

    #     for (images, labels) in zip(self.train_x, self.train_y):
            
    #         optimizer_1.zero_grad()
    #         fx = net(images)
                
    #         client_fx = fx.clone().detach().requires_grad_(True)

        pass

    def train_batch(self, step):

        net = self.net

        self.optimizer_1 = torch.optim.Adam(net.parameters(),lr=LR) 

        images = self.train_x[step]
            
        self.optimizer_1.zero_grad()

        # print(images)

        self.fx = net(images)
                
        client_fx = self.fx.clone().detach().requires_grad_(True)

        return client_fx
        pass


    def back_batch(self, dhat_y):
        # optimizer_1 = torch.optim.Adam(net.parameters(),lr=LR) 

        # optimizer_1.zero_grad()

        self.fx.backward(dhat_y)
        
        self.optimizer_1.step() 
        
        return 

        pass
    
    def update_net(self, net):

        self.net =net
        
        return
    
