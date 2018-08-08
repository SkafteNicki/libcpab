# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:02:06 2018

@author: nsde
"""

#%%
from .cpab import cpab
from . import helper
from . import cpab1d
from . import cpab2d
from . import cpab3d
from sys import platform as _platform
from .helper.utility import check_for_gpu as _check_for_gpu
from .helper.utility import check_cuda_support as _check_cuda_support

#%%
# Check if we can use the slow or the fast version of the cpab implementation 
# in 2D and 3D

print(70*'-')
print('Operating system:', _platform)

_gpu = _check_for_gpu() and _check_cuda_support()
if (_platform == "linux" or _platform == "linux2" \
    or _platform == "darwin") and _gpu: 
   print('Using the fast cuda implementation for CPAB in 2D and 3D')
else:
   # Windows 32 or 64-bit or no GPU
   print('Using the slow pure tensorflow implementation for CPAB in 2D and 3D')
print(70*'-')