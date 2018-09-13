# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 09:58:22 2018

@author: nsde
"""

#%%
from .cpab import cpab
from . import torch_funcs
from . import transformer

from sys import platform as _platform

#%%
print(70*'-')
print('Operating system:', _platform)
print('Using pytorch backend')
print(70*'-')
