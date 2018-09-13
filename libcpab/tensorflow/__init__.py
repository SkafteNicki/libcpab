# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:02:06 2018

@author: nsde
"""
#%%
from .cpab import cpab
from . import cpab1d
from . import cpab2d
from . import cpab3d
from . import helper 
from .helper.gpu_support import gpu_support as _gpu_support
from sys import platform as _platform

#%% 
# Print what version we are going to use
print(70*'-')
print('Operating system:', _platform)
print('Using tensorflow backend')

gpu = _gpu_support()
if gpu:
    print('Using the fast cuda implementation for CPAB')
else:
    # Windows 32 or 64-bit or no GPU
    print('Using the slow pure tensorflow implementation for CPAB')
print(70*'-')
