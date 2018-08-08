# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 07:36:43 2018

@author: nsde
"""
#%%
import numpy as np

#%%
def get_constrain_matrix_1D(nc, domain_min, domain_max,
                            valid_outside, zero_boundary, volume_perservation):
    ncx = nc[0]
    rows = ncx - 1
    cols = 2 * ncx
    
    delta = float(1 / ncx)
    
    L = np.zeros((rows, cols))
    
    for i in range(rows):
        L[i][2*i] = (i+1) * delta
        L[i][2*i+1] = 1
        L[i][2*i+2] = -(i+1) * delta
        L[i][2*i+3] = -1
    
    return L
