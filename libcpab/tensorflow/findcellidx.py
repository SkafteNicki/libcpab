# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:31:34 2018

@author: nsde
"""

#%%
import tensorflow as tf

#%%
def findcellidx(ndim, p, nc):
    if ndim==1:   return findcellidx1D(p, *nc)
    elif ndim==2: return findcellidx2D(p, *nc)
    elif ndim==3: return findcellidx3D(p, *nc)
    
#%%
def findcellidx1D(p, nx):
    pass

#%%
def findcellidx2D(p, nx, ny):
    pass
    
#%%
def findcellidx3D(p, nx, ny, nz):
    pass