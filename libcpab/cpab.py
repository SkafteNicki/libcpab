# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:34:36 2018

@author: nsde
"""

#%%
import numpy as np
from .core.utility import params, get_dir, create_dir, check_if_file_exist, \
                            save_obj, load_obj, null
from .core.setup_constrains_1d import get_constrain_matrix_1D
from .core.setup_constrains_2d import get_constrain_matrix_2D
from .core.setup_constrains_3d import get_constrain_matrix_3D

#%%
class cpab(object):
    """
    
    """
    def __init__(self, 
                 tess_size,
                 backend = 'numpy',
                 zero_boundary=True,
                 volume_perservation=False):
        # Check input
        self._check_input(tess_size, backend, zero_boundary, volume_perservation)
        
        # Parameters
        self.params = params()
        self.params.nc = tess_size
        self.params.ndim = len(tess_size)
        self.params.Ashape = [self.params.ndim, self.params.ndim+1]
        self.params.valid_outside = not(zero_boundary)
        self.params.zero_boundary = zero_boundary
        self.params.volume_perservation = volume_perservation
        self.params.domain_max = [1 for e in self.params.nc]
        self.params.domain_min = [0 for e in self.params.nc]
        self.params.inc = [(self.params.domain_max[i] - self.params.domain_min[i]) / 
                           self.params.nc[i] for i in range(self.params.ndim)]
        self.params.nstepsolver = 50
        
        # For saving the basis
        self._dir = get_dir(__file__) + '/../basis_files/'
        self._basis_file = self._dir + \
                            'cpab_basis_dim' + str(self.params.ndim) + '_tess' + \
                            '_'.join([str(e) for e in self.params.nc]) + '_' + \
                            'vo' + str(int(self.params.valid_outside)) + '_' + \
                            'zb' + str(int(self.params.zero_boundary)) + '_' + \
                            'vp' + str(int(self.params.volume_perservation))
        create_dir(self._dir)
        
        # Specific for the different dims
        if self.params.ndim == 1:
            self.params.nC = self.params.nc[0]
            get_constrain_matrix_f = get_constrain_matrix_1D     
        elif self.params.ndim == 2:
            self.params.nC = 4*np.prod(self.params.nc)
            get_constrain_matrix_f = get_constrain_matrix_2D
        elif self.params.ndim == 3:
            self.params.nC = 6*np.prod(self.params.nc)
            get_constrain_matrix_f = get_constrain_matrix_3D
        
        # Check if we have already created the basis
        if not check_if_file_exist(self._basis_file+'.pkl'):
            # Get constrain matrix
            L = get_constrain_matrix_f(self.params.nc, 
                                       self.params.domain_min, 
                                       self.params.domain_max,
                                       self.params.valid_outside, 
                                       self.params.zero_boundary,
                                       self.params.volume_perservation)
                
            # Find null space of constrain matrix
            B = null(L)
            self.params.constrain_mat = L
            self.params.basis = B
            self.params.D, self.params.d = B.shape
            
            # Save basis as pkl file
            obj = {'basis': self.params.basis, 'constrains': self.params.constrain_mat, 
                   'ndim': self.params.ndim, 'D': self.params.D, 'd': self.params.d, 
                   'nc': self.params.nc, 'nC': self.params.nC, 'Ashape': self.params.Ashape, 
                   'nstepsolver': self.params.nstepsolver}
            save_obj(obj, self._basis_file)
            save_obj(obj, self._dir + 'current_basis')
        
        else: # if it exist, just load it and save as current basis
            file = load_obj(self._basis_file)
            self.params.constrain_mat = file['constrains']
            self.params.basis = file['basis']        
            self.params.D = file['D']
            self.params.d = file['d']
            save_obj(file, self._dir + 'current_basis')
            
        # Load backend
        if backend == 'numpy':
            from .numpy import functions as backend
        elif backend == 'tensorflow':
            from .tensorflow import functions as backend
        elif backend == 'pytorch':
            from .pytorch import functions as backend
        self.backend = backend
        
    #%%
    def get_theta_dim(self):
        """ """
        return self.params.d
    
    #%%
    def get_params(self):
        """ """
        return self.params
    
    #%%
    def get_basis(self):
        """ """
        return self.params.basis
    
    #%%    
    def uniform_meshgrid(self, n_points):
        """ """
        return self.backend.uniform_meshgrid(self.params.ndim,
                                             self.params.domain_min,
                                             self.params.domain_max,
                                             n_points)
      
    #%%
    def sample_transformation(self, n_sample=1, mean=None, cov=None):
        """ """
        if not mean: self._check_type(mean)
        if not cov: self._check_type(cov)
        return self.backend.sample_transformation(n_sample, mean, cov)
    
    #%%
    def sample_transformation_with_prior(self, n_sample=1):
        """ """
        raise NotImplementedError
    
    #%%
    def identity(self, n_sample=1, epsilon=0):
        """ """
        return self.backend.identity(self.params.d, n_sample, epsilon)
    
    #%%
    def transform_grid(self, points, theta):
        """ """
        self._check_type(points)
        self._check_type(theta)
        raise NotImplementedError
    
    #%%    
    def interpolate(self, data, grid, outsize):
        """ """
        self._check_type(data)
        self._check_grid(data)
        raise NotImplementedError
    
    #%%
    def transform_data(self, data, theta, outsize):
        """ """
        self._check_type(data)
        self._check_type(theta)
        raise NotImplementedError
    
    #%%
    def _check_input(self, tess_size, backend, zero_boundary, volume_perservation):
        """ """
        assert len(tess_size) > 0 and len(tess_size) <= 3, \
            '''Transformer only supports 1D, 2D or 3D'''
        assert type(tess_size) == list or type(tess_size) == tuple, \
            '''Argument tess_size must be a list or tuple'''
        assert all([type(e)==int for e in tess_size]), \
            '''All elements of tess_size must be integers'''
        assert all([e > 0 for e in tess_size]), \
            '''All elements of tess_size must be positive'''
        assert backend in ['numpy', 'tensorflow', 'pytorch'], \
            '''Unknown backend, choose between 'numpy', 'tensorflow' or 'pytorch' '''
        assert type(zero_boundary) == bool, \
            '''Argument zero_boundary must be True or False'''
        assert type(volume_perservation) == bool, \
            '''Argument volume_perservation must be True or False'''
            
    #%%
    def _check_type(self, x):
        """ """
        assert type(x) in self.backend.atype(), \
            ''' Input has type {0} but expected type {1} ''' % \
            (type(x), self.backend.atype())