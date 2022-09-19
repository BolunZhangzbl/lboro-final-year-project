# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 23:49:20 2021

@author: Bolun Zhang
"""

import numpy as np
import pickle
from scipy.integrate import cumtrapz
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import keras
from keras.models import Sequential, Model
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import keras.backend.tensorflow_backend as tfback
from sklearn import preprocessing
import tensorflow as tf


def _get_available_gpus():
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
tfback._get_available_gpus = _get_available_gpus()


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
""" Deserialize/Unpickle the dataset to be in a python compatible format """
with open("D:/RML2016.10b/RML2016.10b.dat", 'rb') as f:
    Data = pickle.load(f, encoding='latin1')
    
    
""" Splitting the dataset into samples and their corresponding labels """
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Data.keys())))), [1,0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Data[(mod,snr)])
        for i in range(Data[(mod,snr)].shape[0]):
            lbl.append((mod,snr))
X = np.vstack(X)
X = np.asarray(X)
print(X.shape)
scaler = StandardScaler()
X = scaler.fit_transform(X)
lbl = np.asarray(lbl)


""" Create new features """
X_deriv = []
for x in X:
    X_deriv.append(np.gradient(x))
    
X_int = []
for x in X:
    X_int.append(cumtrapz(x, initial=0))

X_combined = []
for x, x_d, x_i in zip(X, X_deriv, X_int):
    X_combined.append(np.concatenate((x, x_d, x_i), axis=0))
    
X_deriv = np.asarray(X_deriv)
X_int = np.asarray(X_int)
X_combined = np.asarray(X_combined)


""" Splitting data into training and test sets """
X_train, X_test, y_train, y_test = train_test_split(X, lbl, test_size=0.3, random_state=0)
X_deriv_train, X_deriv_test, y_deriv_train, y_deriv_test = train_test_split(X_deriv, lbl, test_size=0.3,
                                                                            random_state=0)
X_int_train, X_int_test, y_int_train, y_int_test = train_test_split(X_int, lbl, test_size=0.3, 
                                                                    random_state=0)
X_combined_train, X_combined_test, y_combined_train, y_combined_test = train_test_split(X_combined, 
                                                                                        lbl, test_size=0.3, random_state=0)

del X_deriv
del X_int
del X_combined    

train = [X_train, X_deriv_train, X_int_train, X_combined_train]
test = [X_test, X_deriv_test, X_int_test, X_combined_test]
lbl_train = [y_train, y_deriv_train, y_int_train, y_combined_train]
lbl_test = [y_test, y_deriv_test, y_int_test, y_combined_test]


""" Baseline Classifier """
y_mods_train = np.asarray(y_train)[:,0]
y_snr_train = np.asarray(y_train)[:,1]
y_mods_test = np.asarray(y_test)[:,0]
y_snr_test = np.asarray(y_test)[:,1]

# Lodistic Regression Classifier
X_train, X_test, y_train, y_test = train_test_split(X, lbl, test_size=0.3, random_state=0)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_mods_train)

acc = []
for snr in snrs:
    print("SNR= ", snr)
    test_X_i = X_test[np.where(y_snr_test == str(snr))]
    test_Y_i = y_mods_test[np.where(y_snr_test == str(snr))]
    y_pred = models.predict(test_X_i)
    
    accuracy = accuracy_score(test_Y_i, y_pred)
    acc.append(accuracy)
    print(" Accuracy = ", accuracy)
    conf = confusion_matrix(test_Y_i, y_pred)
    plt.figure()
    plot_confusion_matrix(conf, labels=mods, title="Logistic Regression Confusion Matrix (SNR=%d)"%(snr))
    
    

