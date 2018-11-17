# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 16:01:52 2018

@author: nsde
"""

#%%
import torch

#%%
def atype():
    return [torch.Tensor]

#%%
def sample_transformation(n_sample=1, mean=None, cov=None):
    pass

#%%
def identity(d, n_sample=1, epsilon=0):
    return torch.zeros(n_sample, d, dtype=torch.float32) + epsilon