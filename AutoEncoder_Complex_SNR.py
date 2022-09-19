# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 00:32:29 2021

@author: Bolun Zhang
"""

from AWGN_ComplexChannel import AutoEncoder_C
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt


EbN0dB_low = -4
EbN0dB_high = 8.5
EbN0dB_num = 26
M = 4
n_channel = 2
k = 2
emb_k = 4
EbN0dB_train = 7
train_data_size = 10000
bertest_data_size = 50000
number = 7

Train_EbN0dB_range = list(np.linspace(start = 10, stop = 20, num = number))
EbN0dB_range = list(np.linspace(start = EbN0dB_low, stop = EbN0dB_high, num = EbN0dB_num))
for (i, train_ebn0db) in enumerate(Train_EbN0dB_range):
    print('train_ebn0db: ', train_ebn0db)
    model_complex = AutoEncoder_C(ComplexChannel = True, CodingMeth = 'Embedding',
                                  M = M, n_channel = n_channel, k = k,
                                  emb_k = emb_k, EbNodB_train = train_ebn0db,
                                  train_data_size = train_data_size)
    model_complex.Initialize()
    model_complex.Cal_BLER(bertest_data_size = bertest_data_size, EbNodB_low = EbN0dB_low,
                           EbNodB_high = EbN0dB_high, EbNodB_num = EbN0dB_num)
    plt.plot(EbN0dB_range, model_complex.ber, '-.', label = 'Train_SNR:%f' % (train_ebn0db))
    
plt.legend(loc = 'lower left')
plt.yscale('log')
plt.xlabel("SNR_RANGE")
plt.ylabel('Block Error Rate')
plt.title('AWGN Channel(2,2), Energy Constraint, SNR Comparison')
plt.grid()

fig = plt.gcf()
fig.set_size_inches(12,10)
plt.show()

