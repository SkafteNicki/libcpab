#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 09:17:19 2018

@author: nsde
"""
#%%
from torch.utils.cpp_extension import load

#%%
square_cuda = load(name = 'square_cuda',
                   sources = ['square_cuda.cpp', 'square_cuda_kernel.cu'], 
                   verbose=False,
                   extra_include_paths = ['/usr/local/cuda-8.0/lib64/'])