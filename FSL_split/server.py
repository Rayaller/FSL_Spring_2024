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

import client
import fed_server

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

#===================================================================
program = "SFLV1 Minist on CNN"
print(f"---------{program}----------")              

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))     

#===================================================================
# parameter setting
#===================================================================
# No. of users
NUM_USERS = 5
EPOCH = 5
frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
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
#client side model difination
#===================================================================

class CNN_server(nn.Module):
    
    def __init__(self):
        pass
        super(CNN_server, self).__init__()

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
        x = self.conv2(x)
        x = x.view(x.size(0), -1) #batch
        # x = x.view(1, -1) # no batch
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
# Data separate
#===================================================================

train_x = []
train_y = []

for _ in range(NUM_USERS):
    train_x_split = []
    train_x.append(train_x_split)

for _ in range(NUM_USERS):
    train_y_split = []
    train_y.append(train_y_split)
 
for step,(batch_x,batch_y) in enumerate(train_loader):
    # print(step)
    train_x[step % NUM_USERS].append(batch_x)
    train_y[step % NUM_USERS].append(batch_y)

# print (np.array(train_x[0]).shape)
# print (np.array(train_y[0]).shape)
    
# print(train_y)

#===================================================================
# training part
#===================================================================
# cnn_client = CNN_client()
cnn_server = CNN_server()
# optimizer = torch.optim.Adam(cnn.parameters(),lr=LR) # 定义优化器

# cnn_client.train()
cnn_server.train()

net_glob_client = CNN_client()
net_glob_client.train()
w_glob_client = net_glob_client.state_dict()

# optimizer_1 = torch.optim.Adam(cnn_client.parameters(),lr=LR) 
optimizer_2 = torch.optim.Adam(cnn_server.parameters(),lr=LR) 
loss_func = nn.CrossEntropyLoss() # 定义损失函数


# class server(object):

clients =[]

for index in range(NUM_USERS):
    client_sig = client.Client(index, train_x, train_y, test_x, test_y) 
    clients.append(client_sig)

total_step = len(train_x[0])

print (total_step)

for epoch in range(EPOCH):
    
    w_locals_client = []
        
    for index in range(NUM_USERS):
            
        for step in range(total_step):

            # client
            hat_y = clients[index].train_batch(step)

            # server
            batch_y = train_y[index][step]
            pred_y = cnn_server(hat_y)
            # print(batch_y)
            # print(pred_y)
            loss = loss_func(pred_y, batch_y)
            optimizer_2.zero_grad()
            loss.backward() # 反向传播
            dhat_y = hat_y.grad.clone().detach()
            optimizer_2.step() # 更新优化器的学习率，一般按照epoch为单位进行更新

            # client
            clients[index].back_batch(dhat_y)
        
         # client
        hat_test_output = clients[index].net(test_x)

        # server
        test_output = cnn_server(hat_test_output)
        pred_y = torch.max(test_output, 1)[1].numpy()
        acc = torch.eq(torch.from_numpy(pred_y), test_y).sum().int()/len(test_y)
        
        print("==========================================================")
        prGreen('Epoch: {:3d}, Client: {:3d}, Avg Accuracy {:.4f} | Avg Loss {:.4f}'.format(epoch, index, loss.data.numpy(), acc))
        print("==========================================================")


        w_locals_client.append(copy.deepcopy(clients[index].net.state_dict()))
    
    w_glob_client = fed_server.FedAvg(w_locals_client)

    net_glob_client.load_state_dict(w_glob_client)

    for index in range(NUM_USERS):

        clients[index].update_net(net_glob_client)
    

    # client
    hat_test_output = clients[0].net(test_x)

    # server
    test_output = cnn_server(hat_test_output)
    pred_y = torch.max(test_output, 1)[1].numpy()
    acc = torch.eq(torch.from_numpy(pred_y), test_y).sum().int()/len(test_y)
    
    print("==========================================================")
    prRed('Epoch: {:3d}, Global, Avg Accuracy {:.4f} | Avg Loss {:.4f}'.format(epoch, loss.data.numpy(), acc))
    print("==========================================================")


# for epoch in range(EPOCH):
 
#     for step,(batch_x,batch_y) in enumerate(train_loader):
#         # client
#         client_y = cnn_client(batch_x)
#         hat_y = client_y.clone().detach().requires_grad_(True) # to make none leaf point has attribute: grad
        
#         # server
#         pred_y = cnn_server(hat_y)
#         loss = loss_func(pred_y, batch_y)
#         optimizer_2.zero_grad()
#         loss.backward() # 反向传播
#         dhat_y = hat_y.grad.clone().detach()
#         optimizer_2.step() # 更新优化器的学习率，一般按照epoch为单位进行更新

#         # client  
#         optimizer_1.zero_grad()
#         client_y.backward(dhat_y)
#         optimizer_1.step() 

#     # client
#     hat_test_output = cnn_client(test_x)

#     # server
#     test_output = cnn_server(hat_test_output)
#     pred_y = torch.max(test_output, 1)[1].numpy()
#     acc = torch.eq(torch.from_numpy(pred_y), test_y).sum().int()/len(test_y)
    
#     prRed("==========================================================")
#     prGreen(' Epoch: {:3d}, Avg Accuracy {:.4f} | Avg Loss {:.4f}'.format(epoch, loss.data.numpy(), acc))
#     prRed("==========================================================")