#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
# load training data
data = np.genfromtxt("ridgetrain.txt")
# input of training data
x_train = data[:,0]
x_train =  x_train.reshape(x_train.shape[0],1)
# output of training data
y_train = data[:,1] 
y_train = y_train.reshape(y_train.shape[0],1)

# Load test data
data1 = np.genfromtxt("ridgetest.txt")
# input of training data
x_test = data1[:,0]
x_test = x_test.reshape(x_test.shape[0],1)
# output of training data
y_test = data1[:,1] 
y_test = y_test.reshape(y_test.shape[0],1)

# Calculates Kernal matrix from trainiing data
def calculat_kernal(l,bp,x):
    K = np.zeros((l,l)) # Zero matrix of size l*l
    for i in range(l):
        for j in range(l):
            K[i][j] = np.exp(-bp*(np.dot((x_train[i]-x_train[j]).T,(x_train[i]-x_train[j])))) # RBF kernel 
    return K
 
# Generate plots
def plot_function(x,y, col):
    plt.scatter(x,y,color=col)
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")


# Predicts output against test input
def predict_y(t,bp):
    a = np.dot(np.linalg.inv(K+t*np.eye(l)),y_train)

    y_pred = np.zeros((y_test.size,1))  # array for predicted outputs

    for i in range(x_test.size):
        for j in range(l):
            y_pred[i] += a[j]*np.exp(-bp*(np.dot((x_test[i]-x_train[j]).T,(x_test[i]-x_train[j]))))
    
    # Calculate RMSE 
    rmse = np.sqrt(np.dot((y_test-y_pred).T,(y_test-y_pred)) / y_pred.size)
    print("RMSE for t = {} is {}".format(t,float(rmse)))
    
    plot_function(x_test,y_test,'b')
    plot_function(x_test,y_pred,'r')
    plt.tight_layout(h_pad= 1)
    plt.show()
    
    


# calculate Kernel Matrix
l = x_train.shape[0]
bp = 0.1 # bandwidth parameter of RBF kernel
K = calculat_kernal(l,bp,x_train)

# hyperparameter for ridge regression
t = [0.1,1,10,100]
for i in t:
    predict_y(i,bp)


# In[ ]:




