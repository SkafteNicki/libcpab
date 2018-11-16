#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 10:30:44 2018

@author: nsde
"""

#%%
from torch.utils.cpp_extension import load

#%%
square_cuda = load(name = 'square_cpp',
                   sources = ['square.cpp'], 
                   verbose=False)