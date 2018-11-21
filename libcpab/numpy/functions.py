# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 16:01:52 2018

@author: nsde
"""

#%%
import numpy as np
from .interpolation import interpolate

#%%
def to(x): 
    return np.array(x)

#%%
def type():
    return [np.ndarray]

#%%
def pdist(mat):
    norm = np.sum(mat * mat, 1)
    norm = np.reshape(norm, (-1, 1))
    D = norm - 2*np.matmul(mat, mat.T) + norm.T
    return D

#%%
def sample_transformation(d, n_sample=1, mean=None, cov=None):
    mean = np.zeros((d,), dtype=np.float32) if mean is None else mean
    cov = np.eye(d, dtype=np.float32) if cov is None else cov
    samples = np.random.multivariate_normal(mean, cov, size=n_sample)
    return samples

#%%
def identity(d, n_sample=1, epsilon=0):
    assert epsilon>=0, "epsilon need to be larger than 0"
    return np.zeros((n_sample, d), dtype=np.float32) + epsilon

#%%
def uniform_meshgrid(ndim, domain_min, domain_max, n_points):
    lin = [np.linspace(domain_min[i], domain_max[i], n_points[i]) for i in range(ndim)]
    mesh = np.meshgrid(*lin[::-1], indexing='ij')
    grid = np.vstack([array.flatten() for array in mesh[::-1]])
    return grid