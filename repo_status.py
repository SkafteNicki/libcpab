# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:50:53 2018

@author: nsde
"""

# TODO: find and fix the sparse tensor -> tensor consume lots of memory error
# TODO: check the gradient at the identity 
#       only seems to be a problem for the pure transformer case
# TODO: fix the problem in the 1D pure transformer shape mismatch between Trels 
#       and points - shape is maybe changed in in the loop?
# TODO: get demo3 to work
# TODO: get demo4 to work
# TODO: test interpolation for 1D
# TODO: test interpolation for 3D

#%%
if __name__ == '__main__':
    from libcpab import cpab
    T1 = cpab([4,])
    T2 = cpab([4,4])
    T3 = cpab([4,4,4])
    
    
    
    
    
    
    