# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 08:53:00 2017

@author: nsde
"""

#%%
import numpy as np
import scipy.linalg as la
from scipy import transpose, compress

#%%
def null(A, eps = 1e-6):
    """ Find the null space of a matrix
        
    Arguments:
        A: `Matrix` [n,m]. Matrix to find the null space of
        eps: `float` (default: 1e-6). Only singular values below the value of
            eps are used to determine the null space
    
    Output:
        `Matrix` [n,m]. The null space of the input matrix
    """
    u, s, vh = la.svd(A)
    padding = np.max([0, np.shape(A)[-1] - np.shape(s)[0]])
    null_mask = np.concatenate(((s <= eps), np.ones((padding,), dtype=bool)), axis=0)
    null_space = compress(null_mask, vh, axis=0)
    return transpose(null_space)

#%%
def create_grid(minbound, maxbound, nb_points):
    """ Create a 2D grid of points
    
    Arguments:
        minbound `list` where minbound[0] is the lower bound in x direction and
            minbound[1] is the lower bound in y direction
        maxbound: `list` where maxbound[0] is the upper bound in x direction and
            maxbound[1] is the upper bound in y direction
        nb_points: `list` where nb_points[0] are the number of points in x
            direction and nb_points[1] are the number of points in y direction
            
    Output:
        points: `Matrix` [2, nb_points[0]*nb_points[1]]. Matrix with 2D grid
            coordinates
    """
    x = np.linspace(minbound[0], maxbound[0], nb_points[0])
    y = np.linspace(minbound[1], maxbound[1], nb_points[1])
    xx, yy = np.meshgrid(x,y)
    points = np.asarray([xx.flatten(), yy.flatten()])
    return points

#%%
if __name__ == '__main__':
    pass

