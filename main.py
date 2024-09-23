#this is a comarison between linear, DT, RandomForest and knn regressors

import random
import numpy as np
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

#start by creating a random dataset

seed = 100
np.random.seed(seed)
random.seed(seed)

rng = np.random.RandomState(1)# This creates a random number generator object using a seed value of 1
X= np.sort(5*rng.rand(80,1),axis = 0)#generates a 80x1 matrix of random numbers, scales them, and then sorts the values.
y = np.sin(X).ravel()#generate y with sin, then flatten to 1D
y+= X.ravel()
y[::5]+= 3*(0.5-rng.rand(16))# add to evrey 5th element in y, so it wont be true sine
X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.2)
X_axis = np.arange(0.0,5.0,0.01)[:,np.newaxis]# create x_axis, every 0.01
regressors = [(LinearRegression(), "red"),
              (KNeighborsRegressor(), "green"),
            (DecisionTreeRegressor(), "blue"),
            (RandomForestRegressor(), "yellow")]

plt.figure()
plt.scatter(X_train,y_train, s=30,edgecolor ="black", c="orange",label = "data" )

plt.xlabel("data")
plt.ylabel("target")
plt.title("Regression Models Comparison")
plt.legend()
plt.show()
