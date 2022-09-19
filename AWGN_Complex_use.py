# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 23:24:00 2021

@author: Bolun Zhang
"""

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from AWGN_ComplexChannel import AutoEncoder_C
from AutoEncoder_BasicModel import AutoEncoder_R

EbN0dB_train = 0
model_test = AutoEncoder_C(ComplexChannel = True, CodingMeth = 'Embedding',
                           M = 16, n_channel = 7, k = 4, emb_k = 16,
                           EbN0dB_train = EbN0dB_train, train_data_size = 10000)
model_test.Initialize()
print("Initialization of the complex model Finished")
#model_test.Draw_Constellation()
model_test.Cal_BLER(EbN0dB_low = -4, EbN0dB_high = 8.5, EbN0dB_num = 26, bertest_data_size = 50000)
EbN0dB_range = list(np.linspace(-4, 8.5, 26))
plt.plot(EbN0dB_range, model_test.ber, 'b.-', label = 'AE_AWGN_RESHAPE(7,4')


model_real = AutoEncoder_R(CodingMeth = 'Embedding', M = 16, n_channel = 7, k = 4, emb_k = 16,
                           EbN0dB_train = EbN0dB_train, train_data_size = 10000)
model_real.Initialize()
print("Initialization of the real model Finished")
model_real.Cal_BLER(bertest_data_size = 50000, EbN0dB_low = -4, EbN0dB_high = 8.5, EbN0dB_num = 26)
plt.plot(EbN0dB_range, model_real.ber, 'y.-', label = 'AE_AWGN(7,4)')

plt.legend(loc='upper right', fontsize = 12)
plt.ysacle('log')
plt.xlabel('SNR_Range')
plt.ylabel('Block Error Rate')
plt.title('aegnChannel(7,4), EnergyConstraint, EbdB_train: %f' % EbN0dB_train)
plt.grid()

fig = plt.gcf()
fig.set_size_inches(12,10)
plt.show()

