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
    
    # Vertices of tesselation
    vertices = np.linspace(domain_min[0], domain_max[0], ncx+1)
    vertices = np.vstack((vertices, np.ones(len(vertices))))
    vertices = vertices[:,1:-1]
    
    # Construct constrain matrix
    rows = ncx - 1
    cols = 2 * ncx
    L = np.zeros((rows, cols))
    for i,v in enumerate(vertices.T):
        L[i,2*i:2*(i+2)] = [*v, *(-v)]
    
    if zero_boundary:
        Ltemp = create_zero_boundary_constrains(ncx, domain_min[0], domain_max[0])
        L = np.vstack((L, Ltemp))
    
    if volume_perservation:
        Ltemp = create_zero_trace_constrains(ncx)
        L = np.vstack((L, Ltemp))
    
    return L

#%%
def create_zero_boundary_constrains(ncx, domain_min, domain_max):
    Ltemp = np.zeros((2,2*ncx))
    Ltemp[0,:2] = [domain_min, 1]
    Ltemp[1,-2:] = [domain_max, 1]
    return Ltemp

#%%
def create_zero_trace_constrains(ncx):
    Ltemp = np.zeros(shape=(ncx, 2*ncx))
    for c in range(ncx):
        Ltemp[c,2*c] = 1
    return Ltemp

#%%
if __name__ == '__main__':
    pass
