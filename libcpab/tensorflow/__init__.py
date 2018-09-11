# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:02:06 2018

@author: nsde
"""
#%%
from . import cpab1d
from . import cpab2d
from . import cpab3d
from . import helper 

#%% 
# utility function for checking if we are using the fast or the slow version
# for computing the CPAB transformations and its gradients
from sys import platform as _platform
from ..helper.utility import gpu_support as _gpu_support

def tf_cpab_version():
    # Print what version we are going to use
    print(70*'-')
    print('Operating system:', _platform)

    _gpu = _gpu_support()
    if _gpu:
        print('Using the fast cuda implementation for CPAB')
    else:
        # Windows 32 or 64-bit or no GPU
        print('Using the slow pure tensorflow implementation for CPAB')
    print(70*'-')