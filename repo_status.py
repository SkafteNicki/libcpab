# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:50:53 2018

@author: nsde
"""

# TODO: test interpolation for 3D

#%%
if __name__ == '__main__':
    from libcpab.develop.cpab import cpab
    T = cpab([2,2])
    g = T.uniform_meshgrid([100, 100])
    theta = 0.1*T.sample_transformation(3)
    theta = T.identity(3)
    gt = T.transform_grid(g, theta)

    
    
    
    
    
    