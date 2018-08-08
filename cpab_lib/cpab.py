# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:02:06 2018

@author: nsde
"""
#%%
from .cpab1d.setup_constrains import get_constrain_matrix_1D
from .cpab2d.setup_constrains import get_constrain_matrix_2D
from .cpab3d.setup_constrains import get_constrain_matrix_3D

from .helper.utility import get_dir
from .helper.math import null

#%%
class cpab:
    '''
    
    '''
    def __init__(self, tess_size, 
                 zero_boundary=False, 
                 volume_perservation=False):
        # Check input
        assert len(tess_size) > 0 and len(tess_size) <= 3, \
            '''Transformer only support 1D, 2D or 3D'''
        assert type(tess_size) == list or type(tess_size) == tuple, \
            '''Argument tess_size must be a list or tuple'''
        assert all([type(e)==int for e in tess_size]), \
            '''All elements of tess_size must be integers'''
        assert all([e > 0 for e in tess_size]), \
            '''All elements of tess_size must be positive'''
        assert type(zero_boundary) == bool, \
            '''Argument zero_boundary must be True or False'''
        assert type(volume_perservation) == bool, \
            '''Argument volume_perservation must be True or False'''
        
        # Parameters
        self.nc = tess_size
        self.ndim = len(tess_size)
        self.valid_outside = not(zero_boundary)
        self.zero_boundary = zero_boundary
        self.volume_perservation = volume_perservation
        self.domain_max = [1 for e in self.nc]
        self.domain_min = [-1 for e in self.nc]
        self.inc = [(self.domain_max[i] - self.domain_min[i]) / 
                    self.nc[i] for i in range(self.ndim)]
        self.basis_name = 'cpab_basis_dim' + str(self.ndim) + '_tess' + \
                          '_'.join([str(e) for e in self.nc]) + '_' + \
                          'vo' + str(int(self.valid_outside)) + '_' + \
                          'zb' + str(int(self.zero_boundary)) + '_' + \
                          'vp' + str(int(self.volume_perservation))
        self.basis_file = get_dir(__file__) + '/../' + self.basis_name
        
        # Get constrain matrix
        if self.ndim == 1:
            L = get_constrain_matrix_1D(self.nc, self.domain_min, self.domain_max,
                                        self.valid_outside, self.zero_boundary,
                                        self.volume_perservation)
        elif self.ndim == 2:
            L = get_constrain_matrix_2D(self.nc, self.domain_min, self.domain_max,
                                        self.valid_outside, self.zero_boundary,
                                        self.volume_perservation)
        elif self.ndim == 3:
            L = get_constrain_matrix_3D(self.nc, self.domain_min, self.domain_max,
                                        self.valid_outside, self.zero_boundary,
                                        self.volume_perservation)
            
        # Find null space of constrain matrix
        B = null(L)
        self.constrains = L
        self.basis = B
        
        
    def transform(self):
        pass
    
    def sample_grid(self):
        pass
    
    def sample_transformation(self):
        pass
