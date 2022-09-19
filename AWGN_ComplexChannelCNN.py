# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 22:01:40 2021

@author: Bolun Zhang
"""

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import numpy as np
import keras
from keras.layers import Input, LSTM, Dense,Conv1D,GaussianNoise, Lambda,Add, Reshape,Dropout, embeddings,Flatten
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from keras.utils.np_utils import to_categorical

# for reproducing result
from numpy.random import seed
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
from numpy import sqrt
from numpy import genfromtxt
from math import pow

from AutoEncoder_BasicModel import AutoEncoder

#set the random state to generate the same/different  train data
from numpy.random import seed
seed(1)
tf.random.set_seed(2)

#tf.compat.v1.enable_eager_execution()
#tf.compat.v1.disable_v2_behavior()


class AutoEncoder_C_CNN(object):
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
        import tensorflow as tf
        seed(1)
        tf.random.set_seed(3)
        assert ComplexChannel in (True, False)
        assert CodingMeth in ('Embedding','Embedding1','Onehot')
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
       H_R = np.random.normal(0, 1, n_sample)
       H_I = np.random.normal(0, 1, n_sample)
       real = H_R * x[:,:,0] - H_I* x[:,:,1]
       imag = H_R * x[:,:,1] + H_I* x[:,:,0]
       print('realshape',K.shape(real))
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
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        
        H_R = np.random.normal(0, 1, n_sample*test_datasize)
        H_I = np.random.normal(0, 1, n_sample*test_datasize)
        H_R = np.reshape(H_R,(-1,2))
        H_I = np.reshape(H_I,(-1,2))
        np.random.shuffle(H_R)
        np.random.shuffle(H_I)
        print('H_R: ', H_R.shape)
        print('H_I: ', H_I.shape)
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
        print('realshape',real.shape)
        print('imagshape',imag.shape)
        #K.clear_session()
        #tf.compat.v1.reset_default_graph()
        x = K.stack([real, imag],axis=2)
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
            train_label = np.random.randint(self.M, size=(self.train_data_size, self.M))
            train_label_out = train_label.reshape((-1, self.M, 1))
            # Embedding Layer
            input_signal = Input(shape=(self.M,))
            encoded = embeddings.Embedding(input_dim=self.M, output_dim=self.emb_k, input_length=self.M)(input_signal)
            encoded1 = Conv1D(filters=self.M, kernel_size=1, activation='relu')(encoded)
            encoded2 = Conv1D(filters=self.n_channel_r, kernel_size=1, activation='linear')(encoded1)
            #encoded3 = LSTM(units=self.n_channel_r, input_shape=(self.M,self.M), return_sequences=True)(encoded2)
            encoded4 = Flatten()(encoded2)
            "fixed power constraint"
            #encoded5 = Lambda(lambda x: np.sqrt(self.n_channel_c) * K.l2_normalize(x, axis=1))(encoded4)
            "average power constraint"
            encoded5 = Lambda(lambda x: x / K.sqrt(K.mean(x**2)))(encoded4)
            encoded6 = Reshape((-1,2))(encoded5)   # Reshape((-1,2))

            channel_out = GaussianNoise(np.sqrt(1 / (2 * self.R * self.EbNo_train)))(encoded6)

            decoded = Reshape((self.M, self.n_channel_r), name='pre_reshape')(channel_out)
            decoded1 = Conv1D(filters=self.M, kernel_size=1, activation='relu', name='pre_receiver')(decoded)
            decoded2 = Conv1D(filters=self.M, kernel_size=1, activation='softmax', name= 'receiver')(decoded1)

            self.auto_encoder = Model(input_signal, decoded2)
            adam = Adam(lr=0.005)
            #rms = RMSprop(lr=0.002)
            self.auto_encoder.compile(optimizer=adam,
                                           loss='sparse_categorical_crossentropy',
                                           metrics=['accuracy'])
            print(self.auto_encoder.summary())
            self.auto_encoder.fit(train_label, train_label_out,
                                       epochs=45,   # 45
                                       batch_size=32,
                                        verbose=1)
            self.encoder = Model(input_signal, encoded6)
            channel_shape = (self.M * self.n_channel_c)
            if self.ComplexChannel == True:
                encoded_input = Input(shape=(channel_shape,2,))
            elif self.ComplexChannel ==False:
                encoded_input = Input(shape=(channel_shape,))
            deco  = self.auto_encoder.layers[-3](encoded_input)
            deco1 = self.auto_encoder.layers[-2](deco)
            deco2 = self.auto_encoder.layers[-1](deco1)
            self.decoder = Model(encoded_input, deco2)
            
        if self.CodingMeth == 'Embedding1':
            print("This model used Embedding layer")
            #Generating train_data
            train_label = np.random.randint(self.M, size=(self.train_data_size, self.M))
            train_label_out = train_label.reshape((-1, self.M, 1))
            # Embedding Layer
            input_signal = Input(shape=(self.M,))
            encoded = embeddings.Embedding(input_dim=self.M, output_dim=self.emb_k, input_length=self.M)(input_signal)
            encoded1 = Conv1D(filters=self.M, kernel_size=1, activation='relu')(encoded)
            encoded2 = Conv1D(filters=self.n_channel_r, kernel_size=1, activation='linear')(encoded1)
            encoded3 = LSTM(units=self.n_channel_r, input_shape=(self.M,self.M), return_sequences=True)(encoded2)
            encoded4 = Flatten()(encoded2)
            "fixed power constraint"
            #encoded5 = Lambda(lambda x: np.sqrt(self.n_channel_c) * K.l2_normalize(x, axis=1))(encoded4)
            "average power constraint"
            encoded5 = Lambda(lambda x: x / K.sqrt(K.mean(x**2)))(encoded4)
            encoded6 = Reshape((-1,2))(encoded5)   # Reshape((-1,2))

            channel_out = GaussianNoise(np.sqrt(1 / (2 * self.R * self.EbNo_train)))(encoded6)

            decoded = Reshape((self.M, self.n_channel_r), name='pre_reshape')(channel_out)
            decoded1 = Conv1D(filters=self.M, kernel_size=1, activation='relu', name='pre_receiver')(decoded)
            decoded2 = Conv1D(filters=self.M, kernel_size=1, activation='softmax', name= 'receiver')(decoded1)

            self.auto_encoder = Model(input_signal, decoded2)
            adam = Adam(lr=0.005)
            #rms = RMSprop(lr=0.002)
            self.auto_encoder.compile(optimizer=adam,
                                           loss='sparse_categorical_crossentropy',
                                           metrics=['accuracy'])
            print(self.auto_encoder.summary())
            self.auto_encoder.fit(train_label, train_label_out,
                                       epochs=45,   # 45
                                       batch_size=32,
                                        verbose=1)
            self.encoder = Model(input_signal, encoded6)
            channel_shape = (self.M * self.n_channel_c)
            if self.ComplexChannel == True:
                encoded_input = Input(shape=(channel_shape,2,))
            elif self.ComplexChannel ==False:
                encoded_input = Input(shape=(channel_shape,))
            deco  = self.auto_encoder.layers[-3](encoded_input)
            deco1 = self.auto_encoder.layers[-2](deco)
            deco2 = self.auto_encoder.layers[-1](deco1)
            self.decoder = Model(encoded_input, deco2)

        """
        The code of onehot situation remain unchaged(AWGN)
        """
        if self.CodingMeth == 'Onehot':
            print("This is the model using Onehot")

            # Generating train_data
            train_label = np.random.randint(self.M, size=(self.train_data_size,self.M))
            train_data = []
            for i in train_label:
                temp = np.zeros(self.M)
                temp[i] = 1
                train_data.append(temp)
            train_data = np.array(train_data)

            input_signal = Input(shape=(self.M,))
            encoded = Dense(self.M, activation='relu')(input_signal)
            encoded1 = Dense(self.n_channel, activation='linear')(encoded)
            encoded2 = Lambda(lambda x: np.sqrt(self.n_channel) * K.l2_normalize(x, axis=1))(encoded1)
            """
            K.l2_mormalize 二阶约束（功率约束）
            """
            EbNo_train = 10 ** (self.EbNodB_train / 10.0)
            encoded3 = GaussianNoise(np.sqrt(1 / (2 * self.R * EbNo_train)))(encoded2)

            decoded  = Dense(self.M, activation='relu')(encoded3)
            decoded1 = Dense(self.M, activation='softmax')(decoded)
            self.auto_encoder = Model(input_signal, decoded1)
            adam = Adam(lr=0.01)
            self.auto_encoder.compile(optimizer=adam, loss='categorical_crossentropy')

            print(self.auto_encoder.summary())

            # for tensor board visualization
            # tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
            # traning auto encoder

            self.auto_encoder.fit(train_data, train_data,
                            epochs=45,
                            batch_size=32,
                            verbose = 1)

            # saving keras model
            from keras.models import load_model

            # if you want to save model then remove below comment
            # autoencoder.save('autoencoder_v_best.model')

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
            scatter_plot = scatter_plot.reshape(self.M, 2, 1)
            plt.scatter(scatter_plot[:, 0], scatter_plot[:, 1],label= '%s,(%d, %d), %d'%(self.CodingMeth,self.n_channel, self.k, self.emb_k) )
            plt.legend()
            plt.axis((-2.5, 2.5, -2.5, 2.5))
            plt.grid()
            plt.show()
        if self.n_channel > 2 :
            if self.CodingMeth == 'Embedding':
                x_emb = self.encoder.predict(test_label)
                print("Embedding,High Dimension")
            if self.CodingMeth == 'Onehot':
                x_emb = self.encoder.predict(test_data)
                print("Onehot,High Dimension")

            EbNo_train = 10 ** (self.EbNodB_train / 10.0)
            noise_std = np.sqrt(1 / (2 * self.R * EbNo_train))
            noise = noise_std * np.random.randn(test_data_size, self.n_channel)
            x_emb = x_emb + noise
            X_embedded = TSNE(learning_rate=700, n_components=2, n_iter=35000, random_state=0,
                              perplexity=60).fit_transform(x_emb)
            print(X_embedded.shape)
            X_embedded = X_embedded / 7
            import matplotlib.pyplot as plt
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1],label= '%s,(%d, %d), %d'%(self.CodingMeth,self.n_channel, self.k, self.emb_k))
            # plt.axis((-2.5,2.5,-2.5,2.5))
            plt.legend()
            plt.grid()
            plt.show()

    def Cal_BLER(self, bertest_data_size = 50000, EbNodB_low = -4, EbNodB_high = 8.5, EbNodB_num = 26):
        test_label = np.random.randint(self.M, size=(bertest_data_size, self.M))
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
            nn_bit = bertest_data_size *self.M
            if self.ComplexChannel == True:
                noise = noise_std * np.random.randn(nn, self.n_channel_c * self.M, 2)
            elif self.ComplexChannel == False:
                noise = noise_std * np.random.rand(nn, self.n_channel_c * self.M)
            if self.CodingMeth == 'Embedding':
                encoded_signal = self.encoder.predict(test_label)
                print('Embedding encoded_signal: ', encoded_signal.shape)
            if self.CodingMeth == 'Embedding1':
                encoded_signal = self.encoder.predict(test_label)
                print('Embedding1 encoded_signal: ', encoded_signal.shape)
            if self.CodingMeth == 'Onehot':
                encoded_signal = self.encoder.predict(test_data)
                print('Onehot encoded_signal: ', encoded_signal.shape)
            #encoded_signal = encoded_signal.reshape((50000, 2, 8))
            #final_signal = self.Rayleigh_Channel_test(x=encoded_signal,n_sample=self.n_channel_c,
            #                                         noise_std=noise_std,
            #                                          test_datasize=bertest_data_size)
            final_signal = encoded_signal + noise
            pred_final_signal = self.decoder.predict(final_signal)
            print('pred_final_signal: ',pred_final_signal.shape)
            pred_output = np.argmax(pred_final_signal, axis=2)
            print('pre_output: ', pred_output.shape)
            print('test_label: ', test_label.shape)
            no_errors = (pred_output != test_label)
            no_errors = no_errors.astype(int).sum()
            ber[n] = no_errors / nn_bit
            print('SNR:', EbNodB_range[n], 'BER:', ber[n])
        self.ber = ber

"""
The following codes show how to apply Class AutoEncoder
"""
"""
model_test3 = AutoEncoder(CodingMeth='Embedding',M = 16, n_channel=7, k = 4, emb_k=16,EbNodB_train = 7,train_data_size=10000)
model_test3.Initialize()
print("Initialization Finished")
#model_test3.Draw_Constellation()
model_test3.Cal_BLER(bertest_data_size= 70000)
EbNodB_range = list(np.linspace(-4, 8.5, 26))
plt.plot(EbNodB_range, model_test3.ber,'bo')
plt.yscale('log')
plt.xlabel('SNR_RANGE')
plt.ylabel('Block Error Rate')
plt.grid()
plt.show()
"""


model_test = AutoEncoder_C_CNN(ComplexChannel=True,CodingMeth='Embedding',
                          M=16, n_channel=7, k=4, emb_k=16,
                          EbNodB_train=7, train_data_size=10000)
model_test.Initialize()
model_test0 = AutoEncoder_C_CNN(ComplexChannel=True,CodingMeth='Embedding1',
                          M=16, n_channel=7, k=4, emb_k=16,
                          EbNodB_train=7, train_data_size=10000)
model_test0.Initialize()
print("Initialization of the complex model Finished")
model_test1 = AutoEncoder(CodingMeth='Embedding',M=16, n_channel=7, k=4, emb_k=16, EbNodB_train = 7,train_data_size=10000)
model_test1.Initialize()
print('Initialization of the basic model Finished')
model_test.Cal_BLER(EbNodB_low=-4, EbNodB_high=8.5, EbNodB_num=26, bertest_data_size= 50000)
model_test0.Cal_BLER(EbNodB_low=-4, EbNodB_high=8.5, EbNodB_num=26, bertest_data_size= 50000)
model_test1.Cal_BLER(EbNodB_low=-4, EbNodB_high=8.5, EbNodB_num=26, bertest_data_size= 50000)
EbNodB_range = list(np.linspace(-4, 8.5, 26))
plt.plot(EbNodB_range, model_test.ber,'bo',label='AE_AWGN_CNN(7,4) without LSTM')
plt.plot(EbNodB_range, model_test0.ber, 'ko', label='AE_AWGN_CNN(7,4) with LSTM')
plt.plot(EbNodB_range, model_test1.ber, 'ro', label='AE_AWGN(7,4)')

plt.legend(loc='upper right', ncol=1, fontsize=16)
plt.yscale('log')
plt.xlabel('Eb/No [dB]', fontsize=16)
plt.ylabel('Block Error Rate', fontsize=16)
plt.title('BLER against Eb/No of CNN vs regular AE', fontsize=20)
plt.grid()

fig = plt.gcf()
fig.set_size_inches(12,10)
plt.show()

#model_test.Draw_Constellation()
