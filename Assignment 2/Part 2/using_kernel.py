#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("kmeans_data.txt")

def predict_cluster(x, u):
    d = np.square(x-u.T)
    c = np.argmin(d, axis=1).reshape(-1,1)
    return c

def transform_input(x,z):   
    return np.exp(-0.1*np.sum(np.square(x - x[z,:]), axis=1)).reshape(-1,1)


for i in range(10):
    k = int((np.random.randint(250, size=1)).reshape(()))
    new_data = transform_input(data,k)
    u = new_data[:2,:] 
    c = predict_cluster(new_data,u)
    c1 = (c==1).reshape(c.shape[0])
    c2 = (c==0).reshape(c.shape[0])
    plt.scatter(data[c1,0], data[c1,1], c='r')
    plt.scatter(data[c2,0], data[c2,1], c='g')
    plt.scatter(data[k,0], data[k,1], c='b')
    plt.show()


# In[ ]:




