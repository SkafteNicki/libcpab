# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 09:33:24 2018

@author: nsde
"""

#%%
import torch
import numpy as np
from ..helper.utility import get_dir, create_dir, save_obj, load_obj, check_if_file_exist
from ..helper.math import null
from ..cpab1d.setup_constrains import get_constrain_matrix_1D
from ..cpab2d.setup_constrains import get_constrain_matrix_2D
from ..cpab3d.setup_constrains import get_constrain_matrix_3D

from .helper import torch_repeat_matrix, torch_expm, torch_interpolate_1D
from .helper import torch_findcellidx_1D, torch_findcellidx_2D, torch_findcellidx_3D

#%%
class params:
    pass # just for saving parameters

#%%
class cpab(object):
    '''
    
    '''
    def __init__(self, tess_size, 
                 zero_boundary=True, 
                 volume_perservation=False,
                 return_tf_tensors=False):
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
        
        # Special cases
        assert not(self.params.ndim==3 and not zero_boundary), \
            '''Non zero boundary is not implemented for 3D'''
        
        # For saving the basis
        self._dir = get_dir(__file__) + '/basis_files/'
        self._basis_name = 'cpab_basis_dim' + str(self.params.ndim) + '_tess' + \
                          '_'.join([str(e) for e in self.params.nc]) + '_' + \
                          'vo' + str(int(self.params.valid_outside)) + '_' + \
                          'zb' + str(int(self.params.zero_boundary)) + '_' + \
                          'vp' + str(int(self.params.volume_perservation))
        self._basis_file = self._dir + self._basis_name
        create_dir(self._dir)
        
        # Specific for the different dims
        if self.params.ndim == 1:
            self.params.nC = self.params.nc[0]
            get_constrain_matrix_f = get_constrain_matrix_1D     
            self.findcellidx = torch_findcellidx_1D
        elif self.params.ndim == 2:
            self.params.nC = 4*np.prod(self.params.nc)
            get_constrain_matrix_f = get_constrain_matrix_2D
            self.findcellidx = torch_findcellidx_2D
        elif self.params.ndim == 3:
            self._nC = 6*np.prod(self.params.nc)
            get_constrain_matrix_f = get_constrain_matrix_3D
            self.findcellidx = torch_findcellidx_3D
            
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
            
        else: # if it exist, just load it
            file = load_obj(self._basis_file)
            self.params.basis = file['basis']
            self.params.constrain_mat = file['constrains']
            self.params.D = file['D']
            self.params.d = file['d']
            
    #%%
    def get_theta_dim(self):
        return self.params.d
    
    #%%
    def get_basis(self):
        return self.params.basis
    
    #%%
    def get_params(self):
        return self.params
        
    #%%
    def uniform_meshgrid(self, n_points):
        assert len(n_points) == self.params.ndim, '''n_points needs to be a list 
            equal to the dimensionality of the transformation'''
        ls = [torch.linspace(self.params.domain_min[i], self.params.domain_max[i], 
                             n_points[i]) for i in range(self.params.ndim)]
        mg = torch.meshgrid(ls)
        grid = torch.cat([g.reshape(1,-1) for g in mg], dim=0)
        return grid
        
    #%%
    def sample_transformation(self, n_sample, mean=None, cov=None):
        mean = torch.zeros((self.params.d,)) if mean is None else mean
        cov = torch.eye(self.params.d) if cov is None else cov
        distribution = torch.distributions.MultivariateNormal(mean, cov)
        return distribution.sample((n_sample,))
        
    #%%
    def identity(self, n_sample):
        return torch.zeros((n_sample, self.params.d))
        
    #%%
    def transform_grid(self, points, theta):
        transformed_points = self._cpab_transformer(points, theta)
        return transformed_points
    
    #%%    
    def interpolate(self, data, grid, outsize):
        grid = (grid*2) - 1 # [0,1] domain to [-1,1] domain
        if self.params.ndim == 1:
            interpolated = torch_interpolate_1D(data, grid)
        elif self.params.ndim == 2:
            grid = grid.permute(0,2,1).reshape(data.shape[0], *outsize, 2).permute(0,2,1,3)
            interpolated = torch.nn.functional.grid_sample(data, grid)
        elif self.params.ndim == 3:
            grid = grid.permute(0,2,1).reshape(data.shape[0], *outsize, 3).permute(0,3,2,1,4)
            interpolated = torch.nn.functional.grid_sample(data, grid)
        return interpolated
        
    #%%
    def transform_data(self, data, theta, outsize):
        ''' '''
        points = self.uniform_meshgrid(outsize)
        transformed_points = self.transform_grid(points, theta)
        transformed_data = self.interpolate(data, transformed_points, outsize)
        return transformed_data
   
    #%%
    def _cpab_transformer(self, points, theta):
        # Problem sizes
        n_points = points.shape[1]
        n_theta = theta.shape[0]
        
        # Transform points
        newpoints = torch_repeat_matrix(points, n_theta) # [n_theta, ndim, n_points]
        newpoints = torch.cat([newpoints, torch.ones(n_theta, 1, n_points)], dim=1)
        newpoints = newpoints.permute(0, 2, 1).reshape(-1, self.params.ndim+1)
        newpoints = newpoints[:,:,None] # [n_theta * ndim, ndim+1, 1]
     
        # Get velocity fields
        B = torch.Tensor(self.params.basis)
        Avees = torch.matmul(B, theta.t())
        As = Avees.t().reshape(n_theta*self.params.nC, *self.params.Ashape)
        AsSquare = torch.cat([As, torch.zeros(n_theta*self.params.nC, 
                                              1, self.params.ndim+1)], dim=1)
        
        # Take matrix exponential
        dT = 1.0 / self.params.nstepsolver
        Trels = torch_expm(dT*AsSquare).type(torch.float32)

        # Batch index to add to correct for the batch effect
        batch_idx = self.params.nC * (torch.ones(n_points, n_theta, dtype=torch.int64) * 
                                      torch.arange(n_theta)).t().flatten()
        
        # Intergrate velocity field
        for i in range(self.params.nstepsolver):
            # Find cell index and correct for batching effect
            idx = self.findcellidx(newpoints, *self.params.nc) + batch_idx
            
            # Gather the correct transformations
            Tidx = Trels[idx]
            
            # Transform points
            newpoints = torch.matmul(Tidx, newpoints)
        
        # Reshape to the right format
        newpoints = newpoints[:,:self.params.ndim].squeeze().t()
        newpoints = newpoints.reshape(self.params.ndim, n_theta, n_points).permute(1,0,2)
        return newpoints
            
    #%%
    def _gpu_support():
        return torch.cuda.is_available()
            
#%%