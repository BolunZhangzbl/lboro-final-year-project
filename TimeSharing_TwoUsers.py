# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 22:06:02 2021

@author: Bolun Zhang
"""

import numpy as np
import fec_block as block
import matplotlib.pyplot as plt
from TwoUserBasicModel import Mycallback, TwoUserEncoder

'''ber = np.array([0.139160013571012, 0.118345838849974, 0.0975593523767089, 0.0774154342655597,
                0.0586184574192509, 0.0418923031812634, 0.0278713063196607, 0.0169667338966386, 
                0.00924721373751743, 0.00439033608734222, 0.00175415061789272, 0.000564706106481744,
                0.000138658688812619, 2.42337854663159e-05, 2.76320800168778e-06, 1.84185551109448e-07])'''
ber = np.array([0.139160013571012, 0.118345838849974, 0.0975593523767089, 0.0774154342655597,
                0.0586184574192509, 0.0418923031812634, 0.0278713063196607, 0.0169667338966386,
                0.00924721373751743, 0.00439033608734222, 0.00175415061789272, 0.000564706106481744,
                0.000138658688812619, 2.42337854663159e-05, 2.76320800168778e-06, 1.84185551109448e-07])

mber = np.array([0.48354, 0.43662, 0.3911, 0.34478, 0.29796, 0.25511, 0.21514, 0.18193, 0.14673,
                 0.1195, 0.09315, 0.07354, 0.05507, 0.03475, 0.02292, 0.01411, 0.00822, 0.00444,
                 0.00208, 10.94e-4, 4.69e-4, 1.9e-4, 3.92e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

SNRdB = np.arange(0, 14, 0.5)
EbNodB = list(np.linspace(0, 14, 16))
Pb_uc = block.block_single_error_Pb_bound(3, SNRdB, False)
plt.plot(SNRdB, Pb_uc, 'k', label='Uncoded QPSK(1,1)')
#plt.plot(SNRdB, Pb_uc*2, 'k:', label='Uncoded QPSK(2,2)')
#plt.plot(SNRdB, Pb_uc*4, 'k-.', label='Uncoded QPSK(4,4)')
#plt.plot(EbNodB, ber*6, 'k--', label='16-QAM')

model1 = TwoUserEncoder(M=2, n_channel=1, k=1, emb_k=2, u1_EbNodB_train=7, u2_EbNodB_train=7,
                            train_datasize=10000, alpha=0.5, beta=0.5)
model1.Initialize()
model1.CalBLER(bertest_data_size=50000, EbNodB_low=0, EbNodB_high=14, EbNodB_num=28)
EbNodB_range = list(np.linspace(0, 14, 28))
plt.plot(EbNodB_range, model1.ber, 'ro', label='AE(1,1)')

plt.yscale('log')
plt.ylim([1e-5, 1e0])
plt.xlim([0, 14])
plt.grid()
plt.ylabel('Block Error Rate', fontsize=15)
plt.xlabel('Eb/N0 [dB]', fontsize=15)
plt.title('BLER versus Eb/N0 for two-user interference channel', fontsize=20)
plt.legend(loc='lower left', fontsize=21)
fig = plt.gcf()
fig.set_size_inches(12,10)
plt.show()



