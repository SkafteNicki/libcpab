# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:02:06 2018

@author: nsde
"""
#%%
from .cpab1d.setup_constrains import get_constrain_matrix_1D
from .cpab1d.transformer import tf_cpab_transformer as tf_cpab1d_transformer
from .cpab2d.setup_constrains import get_constrain_matrix_2D
from .cpab2d.transformer import tf_cpab_transformer as tf_cpab2d_transformer
from .cpab3d.setup_constrains import get_constrain_matrix_3D
from .cpab3d.transformer import tf_cpab_transformer as tf_cpab3d_transformer

from .helper.utility import get_dir, save_obj, load_obj, create_dir, check_if_file_exist
from .helper.math import null

import numpy as np

#%%
class cpab:
    '''
    
    '''
    def __init__(self, tess_size, 
                 zero_boundary=False, 
                 volume_perservation=False,
                 save_basis=True):
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
        self._basis_name = 'cpab_basis_dim' + str(self.ndim) + '_tess' + \
                          '_'.join([str(e) for e in self.nc]) + '_' + \
                          'vo' + str(int(self.valid_outside)) + '_' + \
                          'zb' + str(int(self.zero_boundary)) + '_' + \
                          'vp' + str(int(self.volume_perservation))
        self._basis_file = get_dir(__file__) + '/basis/' + self._basis_name
        create_dir(get_dir(__file__) + '/basis/')
        
        # Functions
        if self.ndim == 1:
            self.get_constrain_matrix_f = get_constrain_matrix_1D
            self.transformer_f = tf_cpab1d_transformer
        elif self.ndim == 2:
            self.get_constrain_matrix_f = get_constrain_matrix_2D
            self.transformer_f = tf_cpab2d_transformer
        elif self.ndim == 3:
            self.get_constrain_matrix_f = get_constrain_matrix_3D
            self.transformer_f = tf_cpab3d_transformer
            
        # Check if we have already created the basis
        if not check_if_file_exist(self._basis_file+'.pkl'):
            # Get constrain matrix
            L = self.get_constrain_matrix_f(self.nc, self.domain_min, self.domain_max,
                                            self.valid_outside, self.zero_boundary,
                                            self.volume_perservation)
                
            # Find null space of constrain matrix
            B = null(L)
            self.constrains = L
            self.basis = B
            self.D, self.d = B.shape
            
            # Save basis as pkl file
            if save_basis:
                save_obj({'basis': self.basis,
                          'constrains': self.constrains,
                          'D': self.D,
                          'd': self.d,
                          'nc': self.nc}, self._basis_file)
        else:
            file = load_obj(self._basis_file)
            self.B = file['basis']
            self.constrains = file['constrains']
            self.D = file['D']
            self.d = file['d']
        
    def transform(self, points, theta):
        assert theta.shape[0] == self.d, \
            'Expects theta to have shape N x ' + self.d
        assert points.shape[0] == self.ndim, \
            'Expects a grid of ' + self.ndim + 'd points'
            
        # Call transformer
        newpoints = self.transformer_f(points, theta)
        return newpoints
    
#    def sample_grid(self):
#        pass
#    
    def sample_transformation(self, n_sample):
        return np.random.normal(size=(n_sample, self.d))
