import numpy as np

# Load the data set
X_seen=np.load('E:\Mtech\Intro to ML\HomeWork\Data/X_seen.npy',encoding='bytes',allow_pickle=True) 
# (40 x N_c x D): 40 feature matrices. X_seen[c] is the N_c x D feature matrix of seen class c and Nc is the number of training inputs of that class
Xtest=np.load('E:\Mtech\Intro to ML\HomeWork\Data/Xtest.npy',encoding='bytes',allow_pickle=True)
# (6180, 4096): feature matrix of the test data.
Ytest=np.load('E:\Mtech\Intro to ML\HomeWork\Data/Ytest.npy',encoding='bytes',allow_pickle=True)
# (6180, 1): ground truth labels of the test data
class_attributes_seen=np.load('E:\Mtech\Intro to ML\HomeWork\Data/class_attributes_seen.npy',encoding='bytes',allow_pickle=True)
# (40, 85): 40x85 matrix with each row being the 85-dimensional class attribute vector of a seen class.
class_attributes_unseen=np.load('E:\Mtech\Intro to ML\HomeWork\Data/class_attributes_unseen.npy',encoding='bytes',allow_pickle=True)
# (10, 85): 10x85 matrix with each row being the 85-dimensional class attribute vector of an unseen class.

# Calculating mean of the seen classes

seen_mean = np.zeros((X_seen.shape[0], X_seen[0].shape[1])) # creates mean matrix of dimension 40x4096 for seen classes.
for i in range(0, X_seen.shape[0]):
    seen_mean[i] = (np.mean(X_seen[i], axis=0)).reshape(1, X_seen[0].shape[1]) 
    # it calculates each row of the mean matrix of dimension 40x4096.


# Calculates the similarity
s = np.dot(class_attributes_unseen, class_attributes_seen.T)
sums = np.sum(s, axis=1)

for row_index in range(s.shape[0]):
    s[row_index] = s[row_index]/sums[row_index] # .reshape(s.shape[0],1) # normalizeing of similarity vector

# Calculating mean of unseen classes
mean_unseen = np.dot(s, seen_mean)


# predicting the class of a test case based on euclidean distance
acc = 0.
dist = np.zeros((Ytest.shape[0], mean_unseen.shape[0]))
for i in range(mean_unseen.shape[0]):
    diff = mean_unseen[i] - Xtest
    sq = np.square(diff)
    sq = np.sum(sq,axis=1) # euclidean distance 
    dist[:, i] = sq

y_pred = np.argmin(dist, axis=1) # retuns index of min value
y_pred = y_pred.reshape(y_pred.shape[0],1)
y_pred+=1
acc = 1 - np.count_nonzero(y_pred-Ytest)/float(Ytest.shape[0])
print("Accuracy of the model on given test cases is: {}".format(100*acc))
