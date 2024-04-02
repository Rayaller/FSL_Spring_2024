
# ============================================================================
#SF learning

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

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))
# to get the same result at each training

#===================================================================
program = "SFLV1 Minist on CNN"
print(f"---------{program}----------")              

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))     



#===================================================================
# model difination
#===================================================================
# No. of users
num_users = 5
EPOCH = 5
frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
LR = 0.0001
BATCH_SIZE = 50 # 批量训练时候一次送入数据的size

class CNN(nn.Module):
    
    def __init__(self):
        pass
        super(CNN, self).__init__()

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

        self.conv2 =nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.out = nn.Linear(32*7*7, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        out= self.out(x)

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




#===================================================================
# training part
#===================================================================
cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR) # 定义优化器
loss_func = nn.CrossEntropyLoss() # 定义损失函数
 
for epoch in range(EPOCH):
 
    for step,(batch_x,batch_y) in enumerate(train_loader):
        pred_y = cnn(batch_x)
        
        if step==0:
            prRed(pred_y)
            prGreen(batch_y)
        loss = loss_func(pred_y,batch_y)
        optimizer.zero_grad() # 清空上一层梯度
        loss.backward() # 反向传播
        optimizer.step() # 更新优化器的学习率，一般按照epoch为单位进行更新
 
        # if step % 50 == 0:
        #     test_output = cnn(test_x)
        #     pred_y = torch.max(test_output, 1)[1].numpy()  # torch.max(test_out,1)返回的是test_out中每一行最大的数)
        #                                                         # 返回的形式为torch.return_types.max(
        #                                                         #           values=tensor([0.7000, 0.9000]),
        #                                                         #           indices=tensor([2, 2]))
        #                                                         # 后面的[1]代表获取indices
          
        #     acc = torch.eq(torch.from_numpy(pred_y), test_y).sum().int()/len(test_y)
        #     print('Epoch: ', epoch, '| train loss: %.4f ' % loss.data.numpy(),'| train acc: %.4f' % acc)
        
    test_output = cnn(test_x)
    pred_y = torch.max(test_output, 1)[1].numpy()
    acc = torch.eq(torch.from_numpy(pred_y), test_y).sum().int()/len(test_y)
    
    prRed("==========================================================")
    prGreen(' Epoch: {:3d}, Avg Accuracy {:.4f} | Avg Loss {:.4f}'.format(epoch, loss.data.numpy(), acc))
    prRed("==========================================================")
 
# 打印前十个测试结果和真实结果进行对比
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
        
        
