# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 13:35:06 2021

@author: Bolun Zhang
"""

import os
from math import sqrt
# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(1)   # reproducible

"""Hyper Parameters"""
EPOCH = 50
BATCH_SIZE = 8
LR = 0.001       # learning rate            
USE_CUDA = False

class_num = 16   # (n=7,k=4) m=16
label = torch.range(1, class_num)-1   # tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15.])
train_labels = label.long()           # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
train_labels = (torch.rand(5) * class_num).long()   # tensor([14, 13,  0,  2,  7])
train_data = torch.sparse.torch.eye(class_num).index_select(dim=0, index=train_labels)   # transfer train_labels to one-hot vectors

#torch_dataset = Data.TensorDataset(data_tensor = train_data, target_tensor = train_labels)
torch_dataset = Data.TensorDataset(train_data, train_labels)   # <torch.utils.data.dataset.TensorDataset at 0x206f35532b0>
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 2)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.transmitter = nn.Sequential(   # input shape (2^4=16, 16)
            nn.Linear(in_features=16, out_features=16, bias=True),
            nn.ReLU(inplace=True),          # activation function
            nn.Linear(16,7))
        
        self.receiver = nn.Sequential(
            nn.Linear(7,16),
            nn.ReLU(inplace=True),
            nn.Linear(16,16))
        
    def forward(self, x):
        x = self.transmitter(x)
        
        # Normalization layer norm2(x)^2 = n
        n = (x.norm(dim=-1)[:,None].view(-1,1).expand_as(x))
        x = sqrt(7) * (x/n)
        
        """noise channel layer"""
        train_SNRdB = 7
        train_SNR = 10 ** (train_SNRdB / 10.0)
        n_channel = 7
        k = 4
        R = k/n_channel    # data rate: bit / channel_use
        noise_var = 1 / (2 * R * train_SNR)   # fixed variance
        noise_std = sqrt(noise_var)
        noise = Variable(torch.randn(x.size()) / ((2 * R * train_SNR) ** 0.5))
        x += noise
        x = self.receiver(x)
        return x
    
net = AutoEncoder()
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=LR)   # optimizer all cnn parameters
loss_func = nn.CrossEntropyLoss()

if __name__ == '__main__':
    # training and testing
    for epoch in range(EPOCH):
        for step, (x,y) in enumerate(loader):    # gives batch data, normalize x when iterate train_loader
            b_x = Variable(x)   # batch x
            b_y = Variable(y)   # batch y
            
            output = net(b_x)   # output
            b_y = (b_y.long()).view(-1)
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data)
            print('\n')
            
            
        

