# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 22:12:37 2018

@author: noura

This is a Multiple Linear Regression Example with two variables.
We use the Scikit Learn library to create the Regression model.
"""

import scipy.stats as ss
import numpy as np
from sklearn.linear_model import LinearRegression

n= 500
B0= 5
B1= 2
B2= -1
np.random.seed(1)
X1= ss.uniform.rvs(size= n) *10
X2= ss.uniform.rvs(size= n) *10
y= B0 + B1 * X1 + B2 * X2 + ss.norm.rvs(loc= 0, scale=1, size=n) 

"""stack X1 and X2 into a matrix X"""
X= np.stack([X1,X2],axis=1)

lm = LinearRegression(fit_intercept = True)
lm.fit(X,y)
"""now you can check your model's parameters by typing lm.intercept_, lm.coef_[0], lm.coef_[1]"""

"""                 --------Prediction Example--------
we will create a new data point X0, then use our model to predict its value.
reshape(1,-1) was added here due to a warning we got from the console."""
X0 = np.array([2,4])
pd = lm.predict(X0.reshape(1,-1)) 

"""                 ----Find the R-Squared Statistic----   
The score() function takes the input values X and y (the training set). 
It generates a prediction y hat, produced by the model.
And it compares the y hat with the true outcome values y in the training set.
"""
print("R-squared: ",lm.score(X,y))

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(X[:,0],X[:,1],y,c=y)
ax.set_xlabel("$X1$")
ax.set_ylabel("$X2$")
ax.set_zlabel("$Y$")