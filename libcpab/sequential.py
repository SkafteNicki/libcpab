# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 10:45:48 2019

@author: nsde
"""

#%%
from .cpab import cpab as cpab_core_class

#%%
class SequentialCpab(object):
    def __init__(self, *cpab):
        self.n_trans = len(cpab)
        self.cpab = cpab
        
        # Assert that all cpab classes are valid
        for i in range(self.n_trans):
            assert isinstance(self.cpab[i], cpab_core_class), \
                ''' Class {0} is not a member of the cpab core class '''.format(i)
        
        # Assert that all cpab classes have same dimensionality
        self.ndim = self.cpab[0].params.ndim
        for i in range(1, self.n_trans):
            assert self.ndim == self.cpab[i].params.ndim, \
                ''' Mismatching dimensionality of transformers. Transformer 0
                have dimensionality {0} but transformer {1} have dimensionality
                {2}'''.format(self.ndim, i, self.cpab[i].params.ndim) 
    
    #%%
    def transform_data(self, data, *theta, outsize):
        assert len(theta) == self.n_trans, \
            ''' Number of parametrizations needed are {0}'''.format(self.n_trans)
        
        grid = self.cpab[0].meshgrid(outsize)
        for i in range(self.n_trans):
            grid = self.cpab[i].transform_grid(grid, theta[i])
        
        data_t = self.cpab[-1].interpolate(data, grid, outsize)
        return data_t
            
        