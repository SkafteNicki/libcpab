# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:02:06 2018

@author: nsde
"""
#%%
from .cpab1d.setup_constrains import get_constrain_matrix_1D
from .cpab2d.setup_constrains import get_constrain_matrix_2D
from .cpab3d.setup_constrains import get_constrain_matrix_3D
from .cpab1d.transformer import tf_cpab_transformer_1D
from .cpab2d.transformer import tf_cpab_transformer_2D
from .cpab3d.transformer import tf_cpab_transformer_3D

from .helper.utility import get_dir, save_obj, load_obj, create_dir, check_if_file_exist
from .helper.math import null

import numpy as np

#%%
class cpab:
    '''
    
    '''
    def __init__(self, tess_size, 
                 zero_boundary=True, 
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
        self.Ashape = [self.ndim, self.ndim+1]
        self.valid_outside = not(zero_boundary)
        self.zero_boundary = zero_boundary
        self.volume_perservation = volume_perservation
        self.domain_max = [1 for e in self.nc]
        self.domain_min = [0 for e in self.nc]
        self.inc = [(self.domain_max[i] - self.domain_min[i]) / 
                    self.nc[i] for i in range(self.ndim)]
        self.nstepsolver = 50
        
        # Special cases
        assert not(self.ndim==3 and not zero_boundary), \
            '''Non zero boundary is not implemented for 3D'''
        
        # For saving the basis
        self._basis_name = 'cpab_basis_dim' + str(self.ndim) + '_tess' + \
                          '_'.join([str(e) for e in self.nc]) + '_' + \
                          'vo' + str(int(self.valid_outside)) + '_' + \
                          'zb' + str(int(self.zero_boundary)) + '_' + \
                          'vp' + str(int(self.volume_perservation))
        self._basis_file = get_dir(__file__) + '/basis_files/' + self._basis_name
        create_dir(get_dir(__file__) + '/basis_files/')
        
        # Specific for the different dims
        if self.ndim == 1:
            self.get_constrain_matrix_f = get_constrain_matrix_1D
            #self.transformer_f = tf_cpab_transformer_1D
            self.nC = self.nc[0]
        elif self.ndim == 2:
            self.get_constrain_matrix_f = get_constrain_matrix_2D
            #self.transformer_f = tf_cpab_transformer_2D
            self.n = 4*np.prod(self.nc)
        elif self.ndim == 3:
            self.get_constrain_matrix_f = get_constrain_matrix_3D
            #self.transformer_f = None #tf_cpab_transformer_3D
            self.nC = 6*np.prod(self.nc)
            
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
        else: # if it exist, just load it
            file = load_obj(self._basis_file)
            self.B = file['basis']
            self.constrains = file['constrains']
            self.D = file['D']
            self.d = file['d']
    
    #%%
    def transform(self, points, theta):
        ''' '''
        assert theta.shape[0] == self.d, \
            'Expects theta to have shape N x ' + self.d
        assert points.shape[0] == self.ndim, \
            'Expects a grid of ' + self.ndim + 'd points'
            
        # Call transformer
        newpoints = self.transformer_f(points, theta, tess=self)
        return newpoints
    
    #%%
    def sample_grid(self, n_points):
        ''' '''
        assert len(n_points) == self.ndim, \
            'n_points needs to be a list equal to the dimensionality of the transformation'
        lin_p = [np.linspace(self.domain_min[i], self.domain_max[i], n_points[i])
                for i in range(self.ndim)]
        mesh_p = np.meshgrid(*lin_p)
        grid = np.vstack([array.flatten() for array in mesh_p])
        return grid
    
    #%%
    def sample_transformation(self, n_sample):
        ''' '''
        return np.random.normal(size=(n_sample, self.d))
    
    #%%
    def interpolate(self, data, transformed_points):
        raise NotImplemented
