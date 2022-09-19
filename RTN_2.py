# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:42:00 2021

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
import numpy as np
from AutoEncoder_BasicModel import AutoEncoder

torch.manual_seed(1)   # reproducible

# Hyper Parameters
EPOCH = 100            # train the training data e times
BATCH_SIZE = 250
LR = 0.01              # learning rate
USE_CUDA = False
n_channel = 8
class_num = 16         # (n=7,k=4)  m=16
# label = torch.range(1, class_num)-1
# train_labels = label.long()
# train_labels = (torch.rand(5) * class_num).long()
# train_data = torch.sparse.torch.eye(class_num).index_select(dim=0, index=train_labels)
# torch_dataset = Data.TensorDataset(data_tensor = train_data, target_tensor = train_labels)
train_labels = (torch.rand(10000) * class_num).long()
train_data = torch.sparse.torch.eye(class_num).index_select(dim=0, index=train_labels)
torch_dataset = Data.TensorDataset(train_data, train_labels)

test_labels = (torch.rand(1500) * class_num).long()
test_data = torch.sparse.torch.eye(class_num).index_select(dim=0, index=test_labels)

loader = Data.DataLoader(                          
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    )                  

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.transmitter = nn.Sequential(         # input shape (2^4=16, 16)
            nn.Linear(in_features=16, out_features=16, bias=True),      
            nn.ReLU(inplace = True),              # activation
            nn.Linear(16,n_channel),
        )

        self.reciever = nn.Sequential(
            nn.Linear(n_channel,16),
            nn.ReLU(inplace = True), 
            nn.Linear(16,16),
        )

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

        x = self.reciever(x)
        return x              # return x for visualization

net = AutoEncoder()
print(net)        # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                      # the target label is not one-hotted

if __name__ == '__main__':
    test_data = Variable(test_data)
    # test_labels = Variable(test_labels)
    # training and testing
    training_loss = []
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
                print(loss.data.item())
                training_loss.append(loss.data.item())
    training_loss = np.array(training_loss)
    print(training_loss.shape)
    
    train_loss = []
    for i in range(0, len(training_loss)):
        if (i%2 == 1):
            train_loss.append(training_loss[i])
    train_loss = np.array(train_loss)
    print(train_loss.shape)
    
    """model_test = AutoEncoder(CodingMeth='Onehot',M = 16, n_channel=8, k = 4, emb_k=16,EbNodB_train = 7,train_data_size=10000)
    model_test.Initialize()
    model_test.Cal_Loss()"""

    ''' plot loss curve '''
    import matplotlib.pyplot as plt
    epochs = range(EPOCH)
    plt.figure()
    #plt.plot(epochs, model_test.loss, 'b', label='Autoencoder')
    plt.plot(epochs, train_loss, 'r--', label='Autoencoder + RTN')
    plt.title('Training Loss (8,4)')
    plt.xlabel('Categorical Cross-entropy Loss')
    plt.ylabel('Epochs')
    plt.legend(loc='upper right', ncol=1)
    plt.grid()
    fig = plt.gcf()
    fig.set_size_inches(12,10)
    plt.show()         

    """
    ''' plot BLER curve '''
    n_channel = 8
    k = 4
    R = k/n_channel
    M = class_num
    test_N = 50000
    test_label = np.random.randint(M, size=test_N)
    test_data = []
    for i in test_label:
        temp = np.zeros(M)
        temp[i] = 1
        test_data.append(temp)
    test_data = np.array(test_data)
    
    encoder = net.transmitter()
    decoder = net.reciever()
      
    EbN0dB_range = list(np.linspace(0, 20, 40))
    ber = [None]*len(EbN0dB_range)
    for n in range(0, len(EbN0dB_range)):
        EbN0 = 10.0**(EbN0dB_range[n]/10.0)
        noise_std = np.sqrt(1/(2*R*EbN0))
        noise_mean = 0
        no_errors = 0
        nn = test_N
        noise = noise_std * np.random.randn(nn, n_channel)
        encoded_signal = encoder.predict(test_data)
        final_signal = encoded_signal + noise
        pred_final_signal = decoder.predict(final_signal)
        pred_output = np.argmax(pred_final_signal, axis = 1)
        no_errors = (pred_output != test_label)
        no_errors = no_errors.astype(int).sum()
        ber[n] = no_errors / (2*nn)
        print('SNR: ', EbN0dB_range[n], 'BER: ', ber[n])"""