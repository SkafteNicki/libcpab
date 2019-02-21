# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 10:45:48 2019

@author: nsde
"""

#%%
from .cpab import cpab as cpab_core_class

#%%
class SequentialCpab(object):
    ''' Helper class meant to make it easy to work with a sequence of transformers.
        Main method of the class are the transform_grid() and transform_data() that
        works similar to the methods of the core class.
        Example:
            T1 = cpab([2,2], ...)
            T2 = cpab([4,4], ...)
            theta1 = T1.sample_theta()
            theta2 = T2.sample_theta()
            T = SequentialCpab(T1, T2)
            data_trans = T.transform_data(some_data, theta1, theta2, outputshape)
    '''
    def __init__(self, *cpab):
        self.n_cpab = len(cpab)
        self.cpab = cpab
        
        # Assert that all cpab classes are valid
        for i in range(self.n_trans):
            assert isinstance(self.cpab[i], cpab_core_class), \
                ''' Class {0} is not a member of the cpab core class '''.format(i)
        
        # Assert that all cpab classes have same dimensionality
        self.ndim = self.cpab[0].params.ndim
        for i in range(1, self.n_cpab):
            assert self.ndim == self.cpab[i].params.ndim, \
                ''' Mismatching dimensionality of transformers. Transformer 0
                have dimensionality {0} but transformer {1} have dimensionality
                {2}'''.format(self.ndim, i, self.cpab[i].params.ndim) 
    
    #%%
    def transform_grid(self, grid, *theta):
        # Check shapes of thetas
        self._assert_theta_shape(*theta)
        
        # Transform in sequence
        for i in range(self.n_cpab):
            grid = self.cpab[i].transform_grid(grid, theta[i])
        
        return grid
        
    #%%
    def transform_data(self, data, *theta, outsize):
        # Check shapes of thetas
        self._assert_theta_shape(*theta)
        
        # Transform in sequence
        grid = self.cpab[0].meshgrid(outsize)
        grid_t = self.transform_grid(grid, *theta)
        
        # Interpolate using final grid
        data_t = self.cpab[-1].interpolate(data, grid_t, outsize)
        return data_t
    
    #%%
    def _assert_theta_shapes(self, *theta):
        n_theta = len(theta)
        assert n_theta == self.n_cpab, \
            ''' Number of parametrizations needed are {0}'''.format(self.n_trans)
        batch_size = theta[0].shape[0]
        for i in range(1, n_theta):
            assert batch_size == theta[i].shape[0], ''' Batch size should be the
                same for all theta's '''
    
    #%%
    def __repr__(self):
        for i in range(self.n_cpab):
            print("======= Transformer {1} ======= ".format(i+1))
            print(self.cpab[i])