# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 11:10:41 2021

@author: Bolun Zhang
"""

import numpy as np
import keras
import tensorflow as tf
from keras.layers import Input, LSTM, Dense, GaussianNoise, Lambda, Add, Reshape, Dropout, embeddings, Flatten
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from keras.utils.np_utils import to_categorical

'For reproducing results'
from numpy.random import seed
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
from numpy import sqrt
from numpy import genfromtxt
from math import pow


''' set the random state to generate the same/different train data '''
seed(1)
tf.random.set_seed(2)


class AutoEncoder_C(object):
    """
    This is an API for the use of NN of an end to end communication system,
    AutoEncoder.Initialize():
        Model Building and Training
    Draw_Constellation()
        Constellation Graph of the transmitted signal
    """
    def __init__(self, ComplexChannel = True,CodingMeth = 'Embedding',M = 4,n_channel = 2, k = 2, emb_k=4, EbNodB_train = 7 , train_data_size = 10000):
        """
        :param CodingMeth: 'Embedding' or ' Onehot'
        :param M: The total number of symbol
        :param n_channel: bits of channel
        :param k: int(log(M))
        :param emb_k: output dimension of the first embedding layer if using the CodingMeth 'Embedding'
        :param EbNodB_train: SNR(dB) of the AWGN channel in train process
        :param train_data_size: size of the train data
        """
        seed(1)
        tf.random.set_seed(3)
        assert ComplexChannel in (True, False)
        assert CodingMeth in ('Embedding','Onehot')
        assert M > 1
        assert n_channel > 1
        assert emb_k > 1
        assert k >1
        self.M = M
        self.CodingMeth = CodingMeth
        self.ComplexChannel = ComplexChannel
        self.n_channel = n_channel
        if ComplexChannel== True:
            self.n_channel_r = self.n_channel * 2
            self.n_channel_c = self.n_channel
        if ComplexChannel == False:
            self.n_channel_r = self.n_channel
            self.n_channel_c = self.n_channel
        self.emb_k = emb_k
        self.k = k
        self.R = self.k / float(self.n_channel)
        self.train_data_size = train_data_size
        self.EbNodB_train = EbNodB_train
        self.EbNo_train = 10 ** (self.EbNodB_train / 10.0)
        self.noise_std = np.sqrt(1 / (2 * self.R * self.EbNo_train))


    def Rayleigh_Channel(self, x, n_sample):
       """
       :param x:
       :param n_sample:
       :return:
       """
       print("x's shape: ", x.shape)
       H_R = np.random.normal(0,1, n_sample)
       H_I = np.random.normal(0,1, n_sample)
       real = H_R * x[:,:,0] - H_I* x[:,:,1]
       imag = H_R * x[:,:,1] + H_I* x[:,:,0]
       print('realshape: ', real.shape)
       print('imagshape: ', imag.shape)
       noise_r = K.random_normal(K.shape(real),
                          mean=0,
                          stddev=self.noise_std)
       noise_i = K.random_normal(K.shape(imag),
                          mean=0,
                          stddev=self.noise_std)
       real = Add()([real, noise_r])
       imag = Add()([imag, noise_i])
       x = K.stack([real, imag], axis=2)
       return x

    def Rayleigh_Channel_test(self, x, n_sample, noise_std, test_datasize):
        """
        :param x:
        :param H:
        :return:
        """
        tf.compat.v1.disable_eager_execution()
        
        H_R = np.random.normal(0, 1, n_sample*test_datasize)
        H_I = np.random.normal(0, 1, n_sample*test_datasize)
        H_R = np.reshape(H_R,(-1,self.n_channel_c))   #n_channel=7
        H_I = np.reshape(H_I,(-1,self.n_channel_c))   #n_channel=7
        np.random.shuffle(H_R)
        np.random.shuffle(H_I)
        #x[:,:,0] is the real part of the signal
        #x[:,:,1] is the imag part of the signal
        real = H_R*x[:,:,0] - H_I*x[:,:,1]
        imag = H_R*x[:,:,1] + H_I*x[:,:,0]
        noise_r = K.random_normal(K.shape(real),
                                mean=0,
                                stddev=noise_std)
        noise_i = K.random_normal(K.shape(imag),
                                mean=0,
                                stddev=noise_std)
        real = real+ noise_r
        imag = imag+ noise_i
        print('realshape: ', real.shape)
        print('imagshape: ', imag.shape)
        x = K.stack([real, imag], axis=2)
        with tf.compat.v1.Session() as sess:
            x = sess.run(x)
        return x

    def Initialize(self):
        """
        :return:
        """

        if self.CodingMeth == 'Embedding':
            print("This model used Embedding layer")
            #Generating train_data
            train_data = np.random.randint(self.M, size=self.train_data_size)
            train_data_pre = train_data.reshape((-1,1))
            # Embedding Layer
            input_signal = Input(shape=(1,))
            encoded = embeddings.Embedding(input_dim=self.M, output_dim=self.emb_k, input_length=1)(input_signal)
            encoded1 = Flatten()(encoded)
            encoded2 = Dense(self.M, activation='relu')(encoded1)
            encoded3 = Dense(self.n_channel_r, activation='linear')(encoded2)
            encoded4 = Lambda(lambda x: np.sqrt(self.n_channel_c) * K.l2_normalize(x, axis=1))(encoded3)
            #encoded4 = BatchNormalization(momentum=0, center=False, scale=False)(encoded3)
            #encoded4 = Lambda(lambda x: x / K.sqrt(K.mean(x**2)))(encoded3)
            encoded5 = Reshape((-1,2))(encoded4)
            
            channel_out = Lambda(lambda x: self.Rayleigh_Channel(x, self.n_channel_c))(encoded5)
            #channel_out = GaussianNoise(np.sqrt(1 / (2 * self.R * self.EbNo_train)))(encoded5)
            
            decoded = Flatten()(channel_out)
            decoded1 = Dense(self.M, activation='relu')(decoded)
            decoded2 = Dense(self.M, activation='softmax')(decoded1)

            self.auto_encoder = Model(input_signal, decoded2)
            adam = Adam(lr=0.005)
            #rms = RMSprop(lr=0.002)
            self.auto_encoder.compile(optimizer=adam,
                                           loss='sparse_categorical_crossentropy',
                                           )
            print(self.auto_encoder.summary())
            self.auto_encoder.fit(train_data, train_data_pre,
                                       epochs=100,   #45
                                       batch_size=32,  #32
                                        verbose=1)
            self.encoder = Model(input_signal, encoded5)

            encoded_input = Input(shape=(self.n_channel_c,2,))
            deco = self.auto_encoder.layers[-3](encoded_input)
            deco1 = self.auto_encoder.layers[-2](deco)
            deco2 = self.auto_encoder.layers[-1](deco1)
            self.decoder = Model(encoded_input, deco2)


        """
        The code of onehot situation remain unchaged(AWGN)
        """
        if self.CodingMeth == 'Onehot':
            print("This is the model using Onehot")

            # Generating train_data
            train_data = np.random.randint(self.M, size=self.train_data_size)
            data = []
            for i in train_data:
                temp = np.zeros(self.M)
                temp[i] = 1
                data.append(temp)
            train_data = np.array(data)

            input_signal = Input(shape=(self.M,))
            encoded = Dense(self.M, activation='relu')(input_signal)
            encoded1 = Dense(self.n_channel, activation='linear')(encoded)
            encoded2 = Lambda(lambda x: np.sqrt(self.n_channel) * K.l2_normalize(x, axis=1))(encoded1)
            #encoded2 = BatchNormalization()(encoded1)
            #encoded2 = Lambda(lambda x: x / K.sqrt(K.mean(x**2)))(encoded1)
            """
            K.l2_mormalize 二阶约束（功率约束）
            """
            EbNo_train = 10 ** (self.EbNodB_train / 10.0)
            encoded3 = GaussianNoise(np.sqrt(1 / (2 * self.R * EbNo_train)))(encoded2)

            decoded = Dense(self.M, activation='relu')(encoded3)
            decoded1 = Dense(self.M, activation='softmax')(decoded)
            self.auto_encoder = Model(input_signal, decoded1)
            adam = Adam(lr=0.01)
            self.auto_encoder.compile(optimizer=adam, loss='categorical_crossentropy')

            print(self.auto_encoder.summary())

            # for tensor board visualization
            # tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
            # traning auto encoder

            self.auto_encoder.fit(train_data, train_data,
                            epochs=45,   #45
                            batch_size=32,
                            verbose = 0)

            # making encoder from full autoencoder
            self.encoder = Model(input_signal, encoded2)

            # making decoder from full autoencoder
            encoded_input = Input(shape=(self.n_channel,))

            deco = self.auto_encoder.layers[-2](encoded_input)
            deco = self.auto_encoder.layers[-1](deco)
            self.decoder = Model(encoded_input, deco)

    def Draw_Constellation(self, test_data_size = 1500):
        """
        :param test_data_size: low-dim situation does not use this param, high-dim situation requires test_data_size to be not to big
        :return:
        """
        import matplotlib.pyplot as plt
        test_label = np.random.randint(self.M, size=test_data_size)
        test_data = []
        for i in test_label:
            temp = np.zeros(self.M)
            temp[i] = 1
            test_data.append(temp)
        test_data = np.array(test_data)

        if self.n_channel == 2:
            scatter_plot = []
            if self.CodingMeth == 'Embedding':
                print("Embedding,Two Dimension")
                for i in range(0, self.M):
                    scatter_plot.append(self.encoder.predict(np.expand_dims(i, axis=0)))
                scatter_plot = np.array(scatter_plot)
            if self.CodingMeth == 'Onehot':
                print("Onehot,Two Dimension")
                for i in range(0, self.M):
                    temp = np.zeros(self.M)
                    temp[i] = 1
                    scatter_plot.append(self.encoder.predict(np.expand_dims(temp, axis=0)))
                scatter_plot = np.array(scatter_plot)
            print('\nBefore Reshape: ', scatter_plot.shape)
            if self.CodingMeth == 'Embedding':
                scatter_plot = scatter_plot.reshape(self.M, self.n_channel_r, 1)
            if self.CodingMeth == 'Onehot':
                scatter_plot = scatter_plot.reshape(self.M, self.n_channel_c, 1)
            print('\nAfter Reshape: ', scatter_plot.shape)
            plt.scatter(scatter_plot[:, 0], scatter_plot[:, 1], label= '%s,(%d, %d), %d'%(self.CodingMeth,self.n_channel, self.k, self.emb_k) )
            plt.legend()
            plt.axhline(color='k', lw=2)
            plt.axvline(color='k', lw=2)
            #plt.axis((-2.5, 2.5, -2.5, 2.5))
            plt.grid()
            plt.show()
        if self.n_channel > 2 :
            EbNo_train = 10 ** (self.EbNodB_train / 10.0)
            noise_std = np.sqrt(1 / (2 * self.R * EbNo_train))
            if self.CodingMeth == 'Embedding':
                x_emb = self.encoder.predict(test_label)
                print("Embedding,High Dimension")
                noise = noise_std * np.random.randn(test_data_size, self.n_channel_r)
                noise = noise.reshape((test_data_size, self.n_channel_c, 2))
                
            if self.CodingMeth == 'Onehot':
                x_emb = self.encoder.predict(test_data)
                print("Onehot,High Dimension")
                noise = noise_std * np.random.randn(test_data_size, self.n_channel_c)
           
            x_emb = x_emb + noise
            X_embedded = TSNE(learning_rate=700, n_components=2, n_iter=35000, random_state=0,
                              perplexity=60).fit_transform(x_emb)
            print(X_embedded.shape)
            X_embedded = X_embedded / 7
            import matplotlib.pyplot as plt
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1],label= '%s,(%d, %d), %d'%(self.CodingMeth,self.n_channel, self.k, self.emb_k))
            # plt.axis((-2.5,2.5,-2.5,2.5))
            plt.axhline(color='k', lw=2)
            plt.axvline(color='k', lw=2)
            plt.legend()
            plt.grid()
            plt.show()

    def Cal_BLER(self, bertest_data_size = 50000, EbNodB_low = -4, EbNodB_high = 8.5, EbNodB_num = 26):
        test_label = np.random.randint(self.M, size=bertest_data_size)
        test_data = []
        for i in test_label:
            temp = np.zeros(self.M)
            temp[i] = 1
            test_data.append(temp)
        test_data = np.array(test_data)

        EbNodB_range = list(np.linspace(EbNodB_low, EbNodB_high, EbNodB_num))
        ber = [None] * len(EbNodB_range)
        self.ber = ber
        for n in range(0, len(EbNodB_range)):
            EbNo = 10 ** (EbNodB_range[n] / 10.0)
            noise_std = np.sqrt(1 / (2 * self.R * EbNo))
            noise_mean = 0
            no_errors = 0
            nn = bertest_data_size
            noise = noise_std * np.random.randn(nn, self.n_channel_c)
            if self.CodingMeth == 'Embedding':
                encoded_signal = self.encoder.predict(test_label)
                final_signal = self.Rayleigh_Channel_test(x=encoded_signal,n_sample=self.n_channel_c,
                                                     noise_std=noise_std,
                                                      test_datasize=bertest_data_size)
            if self.CodingMeth == 'Onehot':
                encoded_signal = self.encoder.predict(test_data)
                final_signal = encoded_signal + noise

            pred_final_signal = self.decoder.predict(final_signal)
            #K.clear_session()
            pred_output = np.argmax(pred_final_signal, axis=1)
            print('pred_finalsignalshape', pred_final_signal.shape)
            print('pre_outputshape',pred_output.shape)
            no_errors = (pred_output != test_label)
            #print(no_errors)
            no_errors = no_errors.astype(int).sum()
            ber[n] = no_errors / nn
            print('SNR:', EbNodB_range[n], 'BER:', ber[n])
        self.ber = ber
        
    def Cal_Loss(self):
        loss = self.auto_encoder.history.history['loss']
        self.loss = loss

"""
The following codes show how to apply Class AutoEncoder_C
"""""""
EbN0dB_train = 7
model_test3 = AutoEncoder_C(CodingMeth='Embedding',M=16, n_channel=4, k=4, emb_k=16,
                            EbNodB_train=7,train_data_size=10000)
model_test3.Initialize()
model_test4 = AutoEncoder_C(CodingMeth='Onehot', M=16, n_channel=4, k=4, emb_k=16,
                            EbNodB_train=7, train_data_size=10000)
model_test4.Initialize()
print("Initialization Finished")
model_test3.Cal_BLER(bertest_data_size = 70000)
model_test4.Cal_BLER(bertest_data_size = 70000)
EbNodB_range = list(np.linspace(-4, 8.5, 26))
plt.plot(EbNodB_range, model_test3.ber, 'bo', label = 'Rayleigh')
plt.plot(EbNodB_range, model_test4.ber, 'ro', label = 'AWGN')
plt.yscale('log')
plt.title('Rayleigh Channel(%d,%d) against AWGN Channel(%d,%d)' %(4,4,4,4), fontsize = 20)
plt.xlabel('Eb/N0 [dB]', fontsize = 18)
plt.ylabel('Block Error Rate', fontsize = 18)
plt.legend(loc = 'lower left', ncol = 1, fontsize = 18)
plt.grid()

fig = plt.gcf()
fig.set_size_inches(12,10)
plt.show()"""

#model_test3.Draw_Constellation()
#model_test4.Draw_Constellation()

epochs = range(100)
ber_RTN = np.array([1.5189, 1.3958, 1.3372, 1.2495, 1.0365, 0.7938, 0.6715, 0.5808, 0.5132, 0.4586,
                    0.4032, 0.3458, 0.2941, 0.2497, 0.2125, 0.1557, 0.1340, 0.1163, 0.1014, 0.0892,
                    0.0790, 0.0701, 0.0626, 0.0563, 0.0505, 0.0461, 0.0417, 0.0381, 0.0349, 0.0322,
                    0.0296, 0.0273, 0.0253, 0.0236, 0.0218, 0.0204, 0.0190, 0.0178, 0.0168, 0.0157,
                    0.0147, 0.0140, 0.0131, 0.0123, 0.0117, 0.0111, 0.0104, 0.0099, 0.0095, 0.0090,
                    0.0085, 0.0082, 0.0077, 0.0074, 0.0071, 0.0067, 0.0064, 0.0061, 0.0059, 0.0056,
                    0.0054, 0.0052, 0.0050, 0.0048, 0.0046, 0.0044, 0.0042, 0.0041, 0.0039, 0.0038,
                    0.0036, 0.0035, 0.0034, 0.0032, 0.0031, 0.0030, 0.0029, 0.0028, 0.0027, 0.0026,
                    0.0025, 0.0025, 0.0024, 0.0023, 0.0022, 0.0021, 0.0021, 0.0020, 0.0019, 0.0019,
                    0.0018, 0.0018, 0.0017, 0.0017, 0.0016, 0.0016, 0.0015, 0.0015, 0.0014, 0.0014])
model = AutoEncoder_C(CodingMeth='Embedding',M=256, n_channel=4, k=8, emb_k=256,
                            EbNodB_train=7,train_data_size=10000)
model.Initialize()
model.Cal_Loss()
plt.plot(epochs, model.loss, 'b', label='Autoencoder')
plt.plot(epochs, ber_RTN, 'r--', label='Autoencoder + RTN')
plt.ylabel('Sparse categorical cross-entropy loss', fontsize=16)
plt.xlabel('Training epoch', fontsize=16)
plt.legend(loc='upper right', ncol=1, fontsize=18)
plt.grid()
plt.ylim([0, 1])
plt.xlim([0, 100])
fig = plt.gcf()
fig.set_size_inches(12,10)
plt.show()
       
                