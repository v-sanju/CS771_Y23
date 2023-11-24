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


# This function will calculate mean of unseen classes for different values of lambda hyperparameter.
def meanUnseenClass(u_seen, Aus, As, k):
    W1 = np.dot(As.T, As) + k*(np.eye(As.shape[1]))
    W2 = np.dot(As.T, u_seen)
    W = np.dot(np.linalg.inv(W1), W2)
    mean = np.dot(Aus, W)
    return mean

# This function will predict the class of unseen test cases and accuracy of the model.
# This model uses euclidean distance from class prototype
def predict_label(u, x_test, y_test):
    acc = 0.
    dist = np.zeros((y_test.shape[0], u.shape[0]))
    for i in range(u.shape[0]):
        diff = u[i] - x_test
        sq = np.square(diff)
        sq = np.sum(sq, axis = 1)
        dist[:, i] = sq

    y_pred = np.argmin(dist, axis=1)
    y_pred = y_pred.reshape(y_pred.shape[0],1)
    y_pred+=1
    acc = 1 - np.count_nonzero(y_pred-y_test)/float(y_test.shape[0])
    return acc

# Calculating mean of seen classes
mean_seen = np.zeros((X_seen.shape[0], X_seen[0].shape[1]))
for i in range(0, X_seen.shape[0]):
    mean_seen[i] = (np.mean(X_seen[i], axis=0)).reshape(1, X_seen[0].shape[1])

# Calculating accuracies for different values of hyperoparameter
for k in [0.1, 1, 4, 8, 12, 15, 20, 30]:
    u_unseen = meanUnseenClass(mean_seen, class_attributes_unseen, class_attributes_seen, k)
    acc = predict_label(u_unseen, Xtest, Ytest)
    print("Test accuracy for lamba = {} is {}".format(k,100*acc))