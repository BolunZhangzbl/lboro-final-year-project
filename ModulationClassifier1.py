# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 23:32:08 2021

@author: Bolun Zhang
"""

from sklearn.preprocessing import LabelBinarizer as LB
from sklearn.preprocessing import normalize 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# --------------------
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.core import Reshape, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras import metrics
# --------------------
from pandas import DataFrame as df
# --------------------
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# --------------------
import tarfile
import pickle
import random
import keras
import sys
import gc

""" Deserialize/Unpickle the dataset to be in a python compatible format """
file = open("D:/RML2016.10b/RML2016.10b.dat", 'rb')
Xd = pickle.load(file, encoding = 'bytes')

snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):
            lbl.append((mod,snr))
X = np.vstack(X)
file.close()


""" Create Features Space """
features = {}
# Raw Time Feature
features['raw'] = X[:,0], X[:,1]
# First derivative in time
features['derivative'] = normalize(np.gradient(X[:,0], axis=1)), normalize(np.gradient(X[:,1],axis=1))
features['integral'] = normalize(np.cumsum(X[:,0], axis=1)), normalize(np.cumsum(X[:,1],axis=1))
# All Together Feature Space
def extract_features(*arguments):
    desired = ()
    for arg in arguments:
        desired += features[arg]
    return np.stack(desired, axis=1)

