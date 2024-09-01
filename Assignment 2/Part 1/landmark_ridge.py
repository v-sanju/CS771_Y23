#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

def transform_x(x, z):
    return np.exp(-0.1*np.square(x - z))

def trainModel_and_plot_results(z,l):
    new_train_x = transform_x(x_train, z)
    new_test_x = transform_x(x_test, z)
    w = np.dot(np.linalg.inv(np.dot(new_train_x.T,new_train_x)+ 0.1*np.eye(new_train_x.shape[1])), np.dot(new_train_x.T,y_train))
    new_y_pred = np.dot(new_test_x,w)
    
    # Calculate RMSE 
    rmse = np.sqrt(np.dot((y_test-new_y_pred).T,(y_test-new_y_pred)) / y_test.size)
    print("RMSE for L = {} is {}".format(l,float(rmse)))
    
    # Plot results
    plot_function(x_test,y_test,'b')
    plot_function(x_test,new_y_pred,'r')
    plt.tight_layout(h_pad= 1)
    plt.show()
    
L = [2,5,20,50,100] # landmark points
for l in L:
    z = np.random.choice(x_train.flatten(),l, replace=False)
    trainModel_and_plot_results(z,l)


# In[ ]:




