1.First loaded data set provided for the model training and testing in below variavbles:
	X_seen -> Input matrix of seen data for each of the 40 seen classes (40*Nc*4096) Nc being the no of training 	data points for a class c.
	Xtest -> Input matrix for test cases (6180x4096)
	Ytest -> Class matrix of each unseen case (6180x1)
	class_attributes_seen -> class attribute for seen class (40x85)
	class_attributes_unseen -> class attributr for unseen class (10x85)

2. Calculated mean matrix (40x4096) of the seen classes as per mean formula:
	mean = (1/Nc)*(X_seen of that class).

3. Mean for unseen classes calculted in different ways for convex and regression methods:
	3(a). For Convex Method:
		First we calculated similarity matrix s and regularized it for using as coefficients of convex 			combination.
		Then calculated mean matrix for unseen classes (10x4096) as convex combination of means of seen 		classes.
	3(b). For Regression Method:
		Calculated means of unseen classes based on closed form solution for regularized multivariate linear 		regression.
4. Prediction of lables for unseen data points.
	Calculated accuracy for Convex Method and Regression method using Euclidean distances from class 	prototype.