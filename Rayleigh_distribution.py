# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 23:58:43 2021

@author: Bolun Zhang
"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
import math

"Plan A"
'sns.distplot(random.rayleigh(size=1000), hist=False)'

"Plan B"
'''
sigma = 1
r = np.arange(0, 5.1, 0.1)
pr = (r/sigma**2)*math.exp(-r**2/2*sigma**2)
pr1 = np.vectorize(pr)
plt.plot(r, pr1, 'b')
'''

"Plan C"
from scipy.stats import rayleigh
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

mean, var, skew, kurt = rayleigh.stats(moments='mvsk')

x = np.linspace(rayleigh.ppf(0.000001),
                rayleigh.ppf(0.999999), 100)
ax.plot(x, rayleigh.pdf(x),
       'r-', lw=5, alpha=0.8, label='rayleigh pdf')


plt.grid()
plt.title('Rayleigh Distribution with σ=1', fontsize=20)
plt.ylabel('p(r)', fontsize=16)
plt.xlabel('r', fontsize=16)
plt.xlim([0,5])
fig = plt.gcf()
fig.set_size_inches(12,10)
plt.show()
