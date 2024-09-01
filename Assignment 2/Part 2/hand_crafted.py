#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

def predict_cluster(x, u):
    d = np.square(x-u.T)
    c = np.argmin(d, axis=1).reshape(-1,1)
    return c

def update_mean(x,c):
    u = np.zeros((2, x.shape[1]))
    u[0,:] = np.mean(x[c==0], axis=0)
    u[1,:] = np.mean(x[c==1], axis=0)
    return u

data = np.genfromtxt("kmeans_data.txt")
tr_data = np.sum(np.square(data), axis = 1).reshape(-1,1) #feature transformation
u = tr_data[:2,:] # initializing mean of the clusters 

for i in range(10):
    c = predict_cluster(tr_data,u)
    u = update_mean(tr_data, c)
    p = (c==1).reshape(c.shape[0])
    n = (c==0).reshape(c.shape[0])
    
plt.scatter(data[p,0], data[p,1], c='r')
plt.scatter(data[n,0], data[n,1], c='g')
plt.show()


# In[ ]:




