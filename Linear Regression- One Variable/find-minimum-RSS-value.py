# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:35:25 2018

@author: noura

This program calculates the sum of squared residuals (SSR, also called SSE) 
for all possible slope values within the range -10 and 15.
The goal here is to find the slope value that yields the minimum SSR
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

n=100
beta_0= 5
beta_1= 2
np.random.seed(1)
x = ss.uniform.rvs(size= n) * 10
y = beta_0 + beta_1 * x + ss.norm.rvs(loc= 0, scale= 1, size= n)

rss = []
slopes = np.arange(-10, 15, 0.001)
for slope in slopes:
    """the deviation from the value predicted by the model:
                       y - (beta_0 - slope*x) """
    """find the RSS for all possible values for the slope ranging from -10 to 15"""            
    rss.append(np.sum((y-beta_0 - slope * x )**2))


"""get the smallest RSS"""    
ind_min = np.argmin(rss)
print("The minimum Residual Sum of Squares: ",rss[ind_min])
print("The slope value that yeilds the minimum Residual Sum of Squares: ",slopes[ind_min])
plt.figure()
plt.plot(slopes,rss)
plt.xlabel("slopes")
plt.ylabel("RSS")
plt.savefig("the minimum sum of square residuals.pdf")