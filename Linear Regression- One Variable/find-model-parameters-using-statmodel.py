# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:15:25 2018

@author: noura
In this example, we use statsmodels.api to find the parameters of our regression model
"""

import statsmodels.api as sm

X = sm.add_constant(x)
mod = sm.OLS(y,X) 
est = mod.fit()
print(est.summary())