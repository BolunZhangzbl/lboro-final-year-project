# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:57:12 2021

@author: Bolun Zhang
"""

import numpy as np
import keras
import tensorflow as tf
from keras.layers import Input, LSTM, Dense, GaussianNoise, Lambda, Dropout, embeddings, Flatten, Add, Reshape
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.callbacks import Callback
from sklearn.manifold import TSNE
#import graphviz
import matplotlib.pyplot as plt
from numpy.random import seed


''' Define the dynamic loss weight '''
class Mycallback(Callback):
    
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.epoch_num = 0
        
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_num = self.epoch_num + 1
        loss1 = logs.get('u1_receiver_loss')
        loss2 = logs.get('u2_receiver_loss')
        print("\nepoch %d" %self.epoch_num)
        print("total_loss %f" %logs.get('loss'))
        print("u1_loss %f" %(loss1))
        print("u2_loss %f" %(loss2))
        u1_ls = loss1 / (loss1 + loss2)
        u2_ls = 1 - u1_ls
        K.set_value(self.a, u1_ls)
        K.set_value(self.b, u2_ls)
        #print("alpha %f" %K.get_value(alpha))
        #print("beta %f" %K.get_value(beta))
        print("selfalpha %f" %K.get_value(self.a))
        print("selfbeta %f" %K.get_value(self.b))
        
        
class TwoUserEncoder(object):
    """
    This is an API
    """
    def __init__(self, ComplexChannel = True, M=4, n_channel=2, k=2, emb_k=4, 
                 u1_EbNodB_train=7, u2_EbNodB_train = 7, train_datasize=10000, alpha=0.5, beta=0.5):
        seed(1)
        tf.random.set_seed(3)
        
        assert ComplexChannel in (True, False)
        assert M > 1
        #assert n_channel > 1
        assert emb_k > 1
        self.M = M
        self.ComplexChannel = ComplexChannel
        self.n_channel = n_channel
        self.k = k
        self.emb_k = emb_k
        self.train_datasize = train_datasize
        self.u1_EbNodB_train = u1_EbNodB_train
        self.u2_EbNodB_train = u2_EbNodB_train
        self.u1_EbNo_train = 10 ** (u1_EbNodB_train / 10.0)
        self.u2_EbNo_train = 10 ** (u2_EbNodB_train / 10.0)
        self.R = self.k / float(self.n_channel)
        if ComplexChannel == True:
            self.n_channel_r = self.n_channel * 2
            self.n_channel_c = self.n_channel
        if ComplexChannel == False:
            self.n_channel_r = self.n_channel
            self.n_channel_c = self.n_channel
        self.u1_noise_std = np.sqrt(1 / (2 * self.R * self.u1_EbNo_train))
        self.u2_noise_std = np.sqrt(1 / (2 * self.R * self.u2_EbNo_train))
        self.alpha = K.variable(alpha)
        self.beta = K.variable(beta)
        
    ''' Define the function for mixed AWGN channel '''
    def mixed_AWGN(self, x, User='u1'):
        assert User in ('u1', 'u2')
        signal = x[0]
        interference = x[1]
        if User == 'u1':
            noise = K.random_normal(K.shape(signal), mean=0, stddev=self.u1_noise_std)
        if User == 'u2':
            noise = K.random_normal(K.shape(signal), mean=0, stddev=self.u2_noise_std)
        signal = Add()([signal, interference])
        signal = Add()([signal, noise])
        return signal
        
    def Initialize(self):
        """Generate train and test data"""
        # user 1
        # seed(1)
        train_label_s1 = np.random.randint(self.M, size=self.train_datasize)
        train_label_out_s1 = train_label_s1.reshape((-1,1))
        # user 2
        # seed(2)
        train_label_s2 = np.random.randint(self.M, size=self.train_datasize)
        train_label_out_s2 = train_label_s2.reshape((-1,1))
        
        """Embedding Model for Two User using real signal"""
        "user1's transmitter"
        u1_input_signal = Input(shape=(1, ))
        u1_encoded = embeddings.Embedding(input_dim=self.M, output_dim=self.emb_k, 
                                          input_length=1)(u1_input_signal)
        u1_encoded1 = Flatten()(u1_encoded)
        u1_encoded2 = Dense(self.M, activation='relu')(u1_encoded1)
        u1_encoded3 = Dense(self.n_channel_r, activation='linear')(u1_encoded2)
        "fixed power constraint"
        #u1_encoded4 = Lambda(lambda x: np.sqrt(self.n_channel_c) * K.l2_normalize(x,axis=1))(u1_encoded3)
        #u1_encoded4 = BatchNormalization(momentum=0, center=False, scale=False)(u1_encoded3)
        #u1_encoded4 = BatchNormalization()(u1_encoded3)
        "Average Power Constraint (For BLER)"
        u1_encoded4 = Lambda(lambda x: x / K.sqrt(K.mean(x**2)))(u1_encoded3)
        
        "user2's transmitter"
        u2_input_signal = Input(shape=(1, ))
        u2_encoded = embeddings.Embedding(input_dim=self.M, output_dim=self.emb_k, 
                                          input_length=1)(u2_input_signal)
        u2_encoded1 = Flatten()(u2_encoded)
        u2_encoded2 = Dense(self.M, activation='relu')(u2_encoded1)
        u2_encoded3 = Dense(self.n_channel_r, activation='linear')(u2_encoded2)
        "fixed power constraint"
        #u2_encoded4 = Lambda(lambda x: np.sqrt(self.n_channel_c) * K.l2_normalize(x,axis=1))(u2_encoded3)
        #u2_encoded4 = BatchNormalization(momentum=0, center=False, scale=False)(u2_encoded3)
        #u2_encoded4 = BatchNormalization()(u2_encoded3)
        "Average Power Constraint  (For BLER)"
        u2_encoded4 = Lambda(lambda x: x / K.sqrt(K.mean(x**2)))(u2_encoded3)
        
        "mixed AWGN channels"
        u1_channel_out = Lambda(lambda x: self.mixed_AWGN(x, User='u1'))([u1_encoded4, u2_encoded4])
        u2_channel_out = Lambda(lambda x: self.mixed_AWGN(x, User='u2'))([u2_encoded4, u1_encoded4])
        
        "user1's receiver"
        u1_decoded = Dense(self.M, activation='relu', name='u1_pre_receiver')(u1_channel_out)
        u1_decoded1 = Dense(self.M, activation='softmax', name='u1_receiver')(u1_decoded)
        
        "user2's receiver"
        u2_decoded = Dense(self.M, activation='relu', name='u2_pre_receiver')(u2_channel_out)
        u2_decoded1 = Dense(self.M, activation='softmax', name='u2_receiver')(u2_decoded)
        
        "twouser's Autoencoder"
        self.twouser_autoencoder = Model(inputs = [u1_input_signal, u2_input_signal],
                                         outputs = [u1_decoded1, u2_decoded1])
        
        adam = Adam(lr=0.01)
        self.twouser_autoencoder.compile(optimizer=adam,
                                         loss='sparse_categorical_crossentropy',
                                         loss_weights=[self.alpha, self.beta])
        self.twouser_autoencoder.summary()
        self.twouser_autoencoder.fit([train_label_s1, train_label_s2], 
                                     [train_label_out_s1, train_label_out_s2],
                                     epochs=45,   #200
                                     batch_size=32,
                                     callbacks=[Mycallback(self.alpha, self.beta)],
                                     verbose=1)
        
        "Creating encoder and decoder for user1"
        self.u1_encoder = Model(u1_input_signal, u1_encoded4)
        
        u1_encoded_input = Input(shape=(self.n_channel_r, ))
        u1_deco  = self.twouser_autoencoder.get_layer("u1_pre_receiver")(u1_encoded_input)
        u1_deco1 = self.twouser_autoencoder.get_layer("u1_receiver")(u1_deco)
        self.u1_decoder = Model(u1_encoded_input, u1_deco1)
        
        "Creating encoder and decoder for user2"
        self.u2_encoder = Model(u2_input_signal, u2_encoded4)
        
        u2_encoded_input = Input(shape=(self.n_channel_r, ))
        u2_deco  = self.twouser_autoencoder.get_layer("u2_pre_receiver")(u2_encoded_input)
        u2_deco1 = self.twouser_autoencoder.get_layer("u2_receiver")(u2_deco)
        self.u2_decoder = Model(u2_encoded_input, u2_deco1)
        
        
    def Draw_Constellation(self, N=1500):
        test_label1 = np.random.randint(self.M, size=N)
        test_label2 = np.random.randint(self.M, size=N)
        test_data = []
        for i in test_label1:
            temp = np.zeros(self.M)
            temp[i] = 1
            test_data.append(temp)
        test_data = np.array(test_data)
        
        if self.n_channel == 2:
            u1_scatter_plot = []
            u2_scatter_plot = []
            for i in range(0, self.M):
                u1_scatter_plot.append(self.u1_encoder.predict(np.expand_dims(i, axis=0)))
                u2_scatter_plot.append(self.u2_encoder.predict(np.expand_dims(i, axis=0)))
            u1_scatter_plot = np.array(u1_scatter_plot)
            u2_scatter_plot = np.array(u2_scatter_plot)
            print("Before reshape")
            print("u1_scatter_plot's shape: ", u1_scatter_plot.shape)
            print("u2_scatter_plot's shape: ", u2_scatter_plot.shape)
            u1_scatter_plot = u1_scatter_plot.reshape(self.M, self.n_channel_r, 1) #reshape(M,2,1)
            u2_scatter_plot = u2_scatter_plot.reshape(self.M, self.n_channel_r, 1) #reshape(M,2,1)
            print("\nAfter reshape")
            print("u1_scatter_plot's shape: ", u1_scatter_plot.shape)
            print("u2_scatter_plot's shape: ", u2_scatter_plot.shape)
            plt.scatter(u1_scatter_plot[:,0], 
                        u1_scatter_plot[:,1], 
                        color='r', label='User1,Embedding,(%d,%d),%d'%(self.n_channel, self.k, self.emb_k))
            plt.scatter(u2_scatter_plot[:,0], 
                        u2_scatter_plot[:,1], 
                        color='b', label='User2,Embedding,(%d,%d),%d'%(self.n_channel, self.k, self.emb_k))
            plt.title('average power TwoUser Constellation Diagram,(%d,%d)'%(self.n_channel, self.k), fontsize=18)
            #plt.legend(loc='lower left')
            plt.axhline(color='k', lw=2)
            plt.axvline(color='k', lw=2)
            #plt.axis((-2.5, 2.5, -2.5, 2.5))
            plt.grid()
            plt.show()
        
        if self.n_channel > 2:
            u1_x_emb = self.u1_encoder.predict(test_label1)
            u2_x_emb = self.u2_encoder.predict(test_label2)
            u1_noise = self.u1_noise_std * np.random.randn(N, self.n_channel_r)
            u2_noise = self.u2_noise_std * np.random.randn(N, self.n_channel_r)
            u1_x_emb = u1_x_emb + u1_noise
            u2_x_emb = u2_x_emb + u2_noise
            print('Before TSNE')
            print("u1_x_emb's shape: ", u1_x_emb.shape)
            print("u2_x_emb's shape: ", u2_x_emb.shape)
            u1_X_embedded = TSNE(learning_rate=700, n_components=2, n_iter=35000, random_state=0,
                                 perplexity=60).fit_transform(u1_x_emb)
            u2_X_embedded = TSNE(learning_rate=700, n_components=2, n_iter=35000, random_state=0,
                                 perplexity=60).fit_transform(u2_x_emb)
            print('\nAfter TSNE')
            print("u1_X_embedded's shape: ", u1_X_embedded.shape)
            print("u2_X_embedded's shape: ", u2_X_embedded.shape)
            u1_X_embedded = u1_X_embedded / self.n_channel #u1_X_embedded = u1_X_embedded / 7
            u2_X_embedded = u2_X_embedded / self.n_channel #u2_X_embedded = u2_X_embedded / 7
            plt.scatter(u1_X_embedded[:,0], u1_X_embedded[:,1], color='r', 
                        label='User1 (%d,%d),%d'%(self.n_channel, self.k, self.emb_k))
            plt.scatter(u2_X_embedded[:,0], u2_X_embedded[:,1], color='b',
                        label='User2 (%d,%d),%d'%(self.n_channel, self.k, self.emb_k))
            #plt.axis((-2.5, 2.5, -2.5, 2.5))
            #plt.legend(loc='lower left')
            plt.title("TwoUser Constellation (TSNE),(%d,%d),%d"%(self.n_channel, self.k, self.emb_k))
            plt.axhline(color='k', lw=2)
            plt.axvline(color='k', lw=2)
            plt.grid()
            plt.show()
        
        
    def CalBLER(self, bertest_data_size, EbNodB_low=0, EbNodB_high=14, EbNodB_num=28):
        "Calculating BER for embedding"
        test_label_s1 = np.random.randint(self.M, size=bertest_data_size)
        test_label_out_s1 = test_label_s1.reshape((-1,1))
        test_label_s2 = np.random.randint(self.M, size=bertest_data_size)
        test_label_out_s2 = test_label_s2.reshape((-1,1))
        
        EbNodB_range = list(np.linspace(EbNodB_low, EbNodB_high, EbNodB_num))
        self.ber = [None] * len(EbNodB_range)
        self.u1_ber = [None] * len(EbNodB_range)
        self.u2_ber = [None] * len(EbNodB_range)
        for n in range(0, len(EbNodB_range)):
            EbNo = 10 ** (EbNodB_range[n] / 10.0)
            noise_std = np.sqrt(1 / (2 * self.R * EbNo))
            noise_mean = 0
            no_errors = 0
            nn = bertest_data_size
            noise1 = noise_std * np.random.randn(nn, self.n_channel_r)
            noise2 = noise_std * np.random.randn(nn, self.n_channel_r)
            u1_encoded_signal = self.u1_encoder.predict(test_label_s1)
            u2_encoded_signal = self.u2_encoder.predict(test_label_s2)
            u1_final_signal = u1_encoded_signal + u2_encoded_signal + noise1
            u2_final_signal = u2_encoded_signal + u1_encoded_signal + noise2
            u1_pred_final_signal = self.u1_decoder.predict(u1_final_signal)
            u2_pred_final_signal = self.u2_decoder.predict(u2_final_signal)
            u1_pred_output = np.argmax(u1_pred_final_signal, axis=1)
            u2_pred_output = np.argmax(u2_pred_final_signal, axis=1)
            u1_no_errors = (u1_pred_output != test_label_s1)
            u1_no_errors = u1_no_errors.astype(int).sum()
            u2_no_errors = (u2_pred_output != test_label_s2)
            u2_no_errors = u2_no_errors.astype(int).sum()
            self.u1_ber[n] = u1_no_errors / nn
            self.u2_ber[n] = u2_no_errors / nn
            self.ber[n] = (self.u1_ber[n] + self.u2_ber[n]) / 2
            print('\n')
            print('U1_SNR: ', EbNodB_range[n], 'U1_BER: ', self.u1_ber[n])
            print('U2_SNR: ', EbNodB_range[n], 'U2_BER: ', self.u2_ber[n])
            print('SNR: ', EbNodB_range[n], 'BER: ', self.ber[n])
        
        
"""
The following code shows how to apply class TwoUserEncoder
"""
'''
model_test = TwoUserEncoder(M=16, n_channel=2, k=4, emb_k=16, u1_EbNodB_train=7, u2_EbNodB_train=7,
                            train_datasize=10000, alpha=0.5, beta=0.5)
model_test.Initialize()
model_test.CalBLER(bertest_data_size=50000, EbNodB_low=0, EbNodB_high=14, EbNodB_num=28)
EbNodB_range = list(np.linspace(0, 14, 28))
#plt.plot(EbNodB_range, model_test.ber, 'bo', label='Embedding, TwoUsers')
plt.plot(EbNodB_range, model_test.u1_ber, 'bo', label='User1')
plt.plot(EbNodB_range, model_test.u2_ber, 'ro', label='User2')
plt.plot(EbNodB_range, model_test.ber, 'k-', label='TwoUsers')
plt.yscale('log')
plt.xlabel('SNR Range: ', fontsize=18)
plt.ylabel('Block Error Rate: ', fontsize=18)
plt.title('TwoUsers BLER (2,4)', fontsize=20)
plt.legend(loc='lower left', ncol=1, fontsize=18)
plt.grid()

fig = plt.gcf()
fig.set_size_inches(12,10)
plt.show()

model_test.Draw_Constellation()
'''

model_test = TwoUserEncoder(M=16, n_channel=2, k=4, emb_k=16, u1_EbNodB_train=7, u2_EbNodB_train=7,
                            train_datasize=10000, alpha=0.5, beta=0.5)
model_test.Initialize()
model_test.Draw_Constellation()
#    import matplotlib.pyplot as plt
#    test_labels = torch.linspace(0, CHANNEL_SIZE-1, steps=CHANNEL_SIZE).long()
#    test_data = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=test_labels)
#    #test_data=torch.cat((test_data, test_data), 1)
#    test_data=Variable(test_data)
#    x=model.encode_signal1(test_data)
#    x = (n_channel**0.5) * (x / x.norm(dim=-1)[:, None])
#    plot_data=x.data.numpy()
#    plt.scatter(plot_data[:,0],plot_data[:,1],color='r')
#    plt.axis((-2.5,2.5,-2.5,2.5))
#    #plt.grid()
#
#    scatter_plot = []
#
#    scatter_plot = np.array(scatter_plot)
#    print (scatter_plot.shape)
#
#    test_labels = torch.linspace(0, CHANNEL_SIZE-1, steps=CHANNEL_SIZE).long()
#    test_data = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=test_labels)
#    #test_data=torch.cat((test_data, test_data), 1)
#    test_data=Variable(test_data)
#    x=model.encode_signal2(test_data)
#    x = (n_channel**0.5) * (x / x.norm(dim=-1)[:, None])
#    plot_data=x.data.numpy()
#    plt.scatter(plot_data[:,0],plot_data[:,1])
#    plt.axis((-2.5,2.5,-2.5,2.5))
#    plt.grid()
#   # plt.show()
#    scatter_plot = []
##
##    scatter_plot = np.array(scatter_plot)
#    plt.show()
    