# -*- coding: utf-8 -*-
"""
Created on Mon May 20 20:11:42 2019

@author: Kevin
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.svm import SVR
regressor = SVR(kernal = 'rbf')

y_pred = regressor.predict(6.5)

plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()