# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 10:26:30 2018

@author: nsde
"""

#%%
from scipy.interpolate import LinearNDInterpolator

#%%
def interpolate(ndim, data, grid, outsize):
    if ndim==1: return interpolate1D(data, grid, outsize)
    elif ndim==2: return interpolate2D(data, grid, outsize)
    elif ndim==3: return interpolate3D(data, grid, outsize)

#%%    
def interpolate1D(data, grid, outsize):
    for d, g in zip(data, grid):
        inter_f = LinearNDInterpolator(g, d)
    return inter_f(g)

#%%    
def interpolate2D(data, grid, outsize):
    for d, g in zip(data, grid):
        inter_f = LinearNDInterpolator(g, d)
    return inter_f(g)

#%%    
def interpolate3D(data, grid, outsize):
    for d, g in zip(data, grid):
        inter_f = LinearNDInterpolator(g, d)
    return inter_f(g)