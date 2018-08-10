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
from .helper.tf_interpolate import tf_interpolate_1D, tf_interpolate_2D, tf_interpolate_3D
from .helper.tf_findcellidx import tf_findcellidx_1D, tf_findcellidx_2D, tf_findcellidx_3D

from .helper.utility import get_dir, save_obj, load_obj, create_dir, check_if_file_exist
from .helper.math import null

import numpy as np
import tensorflow as tf

#%%
class cpab(object):
    '''
    
    '''
    def __init__(self, tess_size, 
                 zero_boundary=True, 
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
        self._dir = get_dir(__file__) + '/basis_files/'
        self._basis_name = 'cpab_basis_dim' + str(self.ndim) + '_tess' + \
                          '_'.join([str(e) for e in self.nc]) + '_' + \
                          'vo' + str(int(self.valid_outside)) + '_' + \
                          'zb' + str(int(self.zero_boundary)) + '_' + \
                          'vp' + str(int(self.volume_perservation))
        self._basis_file = self._dir + self._basis_name
        create_dir(self._dir)
        
        # Specific for the different dims
        if self.ndim == 1:
            self.get_constrain_matrix_f = get_constrain_matrix_1D
            self.transformer_f = tf_cpab_transformer_1D
            self.interpolate_f = tf_interpolate_1D
            self.findcellidx_f = tf_findcellidx_1D
            self.nC = self.nc[0]
        elif self.ndim == 2:
            self.get_constrain_matrix_f = get_constrain_matrix_2D
            self.transformer_f = tf_cpab_transformer_2D
            self.interpolate_f = tf_interpolate_2D
            self.findcellidx_f = tf_findcellidx_2D
            self.nC = 4*np.prod(self.nc)
        elif self.ndim == 3:
            self.get_constrain_matrix_f = get_constrain_matrix_3D
            self.transformer_f = tf_cpab_transformer_3D
            self.interpolate_f = tf_interpolate_3D
            self.findcellidx_f = tf_findcellidx_3D
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
            obj = {'basis': self.basis, 'constrains': self.constrains, 'ndim': self.ndim,
                   'D': self.D, 'd': self.d, 'nc': self.nc, 'nC': self.nC,
                   'Ashape': self.Ashape, 'nstepsolver': self.nstepsolver}
            save_obj(obj, self._basis_file)
            save_obj(obj, self._dir + 'current_basis')
            
        else: # if it exist, just load it
            file = load_obj(self._basis_file)
            self.basis = file['basis']
            self.constrains = file['constrains']
            self.D = file['D']
            self.d = file['d']
            
            # Save as the current basis
            obj = {'basis': self.basis, 'constrains': self.constrains, 'ndim': self.ndim,
                   'D': self.D, 'd': self.d, 'nc': self.nc, 'nC': self.nC,
                   'Ashape': self.Ashape, 'nstepsolver': self.nstepsolver}
            save_obj(obj, self._dir + 'current_basis')
            
        # To run tensorflow
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
            
    #%%
    def transform(self, points, theta):
        ''' '''
        assert theta.shape[1] == self.d, \
            'Expects theta to have shape N x ' + str(self.d)
        assert points.shape[0] == self.ndim, \
            'Expects a grid of ' + str(self.ndim) + 'd points'
            
        # Call transformer
        points = tf.cast(points, tf.float32)
        theta = tf.cast(theta, tf.float32)
        newpoints = self.sess.run(self.transformer_f(points, theta))
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
        # Call interpolator
        interpolate = self.sess.run(self.interpolate_f(data, transformed_points))
        return interpolate
    
    #%%
    def theta2Avees(self, theta):
        Avees = self.basis.dot(theta)
        return Avees
    
    #%%
    def Avees2As(self, Avees):
        As = np.reshape(Avees, (self.nC, *self.Ashape))
        return As
    
    def As2squareAs(self, As):
        squareAs = np.zeros(shape=(self.nC, self.ndim+1, self.ndim+1))
        squareAs[:,:-1,:] = As
        return squareAs
    
    #%%
    def calc_v(self, points, theta):
        # Construct the affine transformations
        Avees = self.theta2Avees(theta)
        As = self.Avees2As(Avees)
        
        # Find cells and extract correct affine transformation
        idx = self.sess.run(self.findcellidx_f(points.T, *self.nc))
        Aidx = As[idx]
        
        # Make homogeneous coordinates
        points = np.expand_dims(np.vstack((points, np.ones((1, points.shape[1])))).T,2)
        
        # Do matrix-vector multiplication
        v = np.matmul(Aidx, points)
        return np.squeeze(v).T
    
    #%%
    def visualize_vector_field(self, theta, nb_points=10):
        points = self.sample_grid([nb_points for i in range(self.ndim)])
        v = self.calc_v(points, theta)
        
        # Plot
        import matplotlib.pyplot as plt
        if self.ndim==1:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.quiver(points[0,:], np.zeros_like(points), v, np.zeros_like(v))
            ax.set_xlim(self.domain_min[0], self.domain_max[0])
        elif self.ndim==2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.quiver(points[0,:], points[1,:], v[0,:], v[1,:])
            ax.set_xlim(self.domain_min[0], self.domain_max[0])
            ax.set_ylim(self.domain_min[1], self.domain_max[1])
        elif self.ndim==3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.quiver(points[0,:], points[1,:], points[2,:], v[0,:], v[1,:], v[2,:],
                      length=0.3, arrow_length_ratio=0.5)
            ax.set_xlim3d(self.domain_min[0], self.domain_max[0])
            ax.set_ylim3d(self.domain_min[1], self.domain_max[1])
            ax.set_zlim3d(self.domain_min[2], self.domain_max[2])
        plt.axis('equal')
        plt.title('Velocity field')
        plt.show()
        