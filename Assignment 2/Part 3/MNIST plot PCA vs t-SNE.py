#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pickle
with open('mnist_small.pkl', 'rb') as file:
    data = pickle.load(file)
x = data['X']
y = data['Y']

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(x)
std_X = scalar.transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(std_X)
x_pca = pca.transform(std_X)

colors = ['b', 'g', 'r', 'c', 'm','y','k','#FFA500','#800080','#008080']
import matplotlib.pyplot as plt
plt.scatter(x_pca[:,0],x_pca[:,1], c = [colors[i] for i in y.T.flatten()] )
plt.title("2D plot of MNIST data using PCA")
plt.show()


# In[66]:


# import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(x)

plt.scatter(X_tsne[:, 0], X_tsne[:,1], c = [colors[i] for i in y.T.flatten()])
plt.title("2D plot of MNIST data using t-SNE")
plt.show()


# In[ ]:




