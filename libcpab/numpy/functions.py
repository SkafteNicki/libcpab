# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 16:01:52 2018

@author: nsde
"""

#%%
import numpy as np

#%%
def atype():
    return [np.ndarray]

#%%
def sample_transformation(n_sample=1, mean=None, cov=None):
    pass

#%%
def identity(d, n_sample=1, epsilon=0):
    return np.zeros((n_sample, d), dtype=np.float32) + epsilon