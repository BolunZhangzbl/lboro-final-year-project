# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:51:06 2021

@author: Bolun Zhang
"""

import os
from math import sqrt
# third party library
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(1)


''' Hyper Parameter '''
EPOCH = 100         
BATCH_SIZE = 250
LR = 0.01
USE_CUDA = False
n_channel = 7
class_num = 16   # (n=7,k=4) m=16


''' Trainind and Testing data '''
train_labels = (torch.rand(10000) * class_num).long()
train_data = torch.sparse.torch.eye(class_num).index_select(dim=0, index=train_labels)
torch_dataset = Data.TensorDataset(train_data, train_labels)

test_labels = (torch.rand(1500) * class_num).long()
test_data = torch.sparse.torch.eye(class_num).index_select(dim=0, index=test_labels)

loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.transmitter = nn.Sequential(
            nn.Linear(in_features=16, out_features=16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(16,n_channel))
        
        self.receiver = nn.Sequential(
            nn.Linear(n_channel,16),
            nn.ReLU(inplace=True),
            nn.Linear(16,16))
        
    def forward(self, x):
        x = self.transmitter(x)

        # Normalization layer norm2(x)^2 = n
        n = (x.norm(dim=-1)[:,None].view(-1,1).expand_as(x))
        x = sqrt(n_channel)*(x / n)

        """channel noise layer""" 
        training_SNR = 5.01187     # 7dBW to SNR. 10**(7/10)
        communication_rate = 4/n_channel   # bit / channel_use
        # 1/(2*communication_rate*training_SNR)   fixed variance
        noise = Variable(torch.randn(x.size()) / ((2*communication_rate*training_SNR) ** 0.5))
        x += noise

        x = self.receiver(x)
        return x              # return x for visualization

net = AutoEncoder()
print(net)        # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                      # the target label is not one-hotted

if __name__ == '__main__':
    test_data = Variable(test_data)
    # test_labels = Variable(test_labels)
    # training and testing
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(loader):   # gives batch data, normalize x when iterate train_loader
            b_x = Variable(x)   # batch x
            b_y = Variable(y)   # batch y

            output = net(b_x)             # output
            b_y = (b_y.long()).view(-1)
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
            if step % 20 == 0:
                test_output = net(test_data)
                pred_labels = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = sum(pred_labels == test_labels) / float(test_labels.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data, '| test accuracy: %.2f' % accuracy)


