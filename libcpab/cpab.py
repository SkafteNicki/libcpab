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
        self._nc = tess_size
        self._ndim = len(tess_size)
        self._Ashape = [self._ndim, self._ndim+1]
        self._valid_outside = not(zero_boundary)
        self._zero_boundary = zero_boundary
        self._volume_perservation = volume_perservation
        self._domain_max = [1 for e in self._nc]
        self._domain_min = [0 for e in self._nc]
        self._inc = [(self._domain_max[i] - self._domain_min[i]) / 
                    self._nc[i] for i in range(self._ndim)]
        self._nstepsolver = 50
        
        # Special cases
        assert not(self._ndim==3 and not zero_boundary), \
            '''Non zero boundary is not implemented for 3D'''
        
        # For saving the basis
        self._dir = get_dir(__file__) + '/basis_files/'
        self._basis_name = 'cpab_basis_dim' + str(self._ndim) + '_tess' + \
                          '_'.join([str(e) for e in self._nc]) + '_' + \
                          'vo' + str(int(self._valid_outside)) + '_' + \
                          'zb' + str(int(self._zero_boundary)) + '_' + \
                          'vp' + str(int(self._volume_perservation))
        self._basis_file = self._dir + self._basis_name
        create_dir(self._dir)
        
        # Specific for the different dims
        if self._ndim == 1:
            self._nC = self._nc[0]
            get_constrain_matrix_f = get_constrain_matrix_1D
            self._transformer = tf_cpab_transformer_1D
            self._interpolate = tf_interpolate_1D
            self._findcellidx = tf_findcellidx_1D            
        elif self._ndim == 2:
            self._nC = 4*np.prod(self._nc)
            get_constrain_matrix_f = get_constrain_matrix_2D
            self._transformer = tf_cpab_transformer_2D
            self._interpolate = tf_interpolate_2D
            self._findcellidx = tf_findcellidx_2D
        elif self._ndim == 3:
            self._nC = 6*np.prod(self._nc)
            get_constrain_matrix_f = get_constrain_matrix_3D
            self._transformer = tf_cpab_transformer_3D
            self._interpolate = tf_interpolate_3D
            self._findcellidx = tf_findcellidx_3D
            
        # Check if we have already created the basis
        if not check_if_file_exist(self._basis_file+'.pkl'):
            # Get constrain matrix
            L = get_constrain_matrix_f(self._nc, self._domain_min, self._domain_max,
                                       self._valid_outside, self._zero_boundary,
                                       self._volume_perservation)
                
            # Find null space of constrain matrix
            B = null(L)
            self._constrain_mat = L
            self._basis = B
            self._D, self._d = B.shape
            
            # Save basis as pkl file
            obj = {'basis': self._basis, 'constrains': self._constrain_mat, 'ndim': self._ndim,
                   'D': self._D, 'd': self._d, 'nc': self._nc, 'nC': self._nC,
                   'Ashape': self._Ashape, 'nstepsolver': self._nstepsolver}
            save_obj(obj, self._basis_file)
            save_obj(obj, self._dir + 'current_basis')
            
        else: # if it exist, just load it
            file = load_obj(self._basis_file)
            self._basis = file['basis']
            self._constrain_mat = file['constrains']
            self._D = file['D']
            self._d = file['d']
            
            # Save as the current basis
            obj = {'basis': self._basis, 'constrains': self._constrain_mat, 'ndim': self._ndim,
                   'D': self._D, 'd': self._d, 'nc': self._nc, 'nC': self._nC,
                   'Ashape': self._Ashape, 'nstepsolver': self._nstepsolver}
            save_obj(obj, self._dir + 'current_basis')
            
        # To run tensorflow
        self._return_tf_tensors = return_tf_tensors
        self._fixed_data = False
        self._sess = tf.Session()
        
    #%%
    def get_theta_dim(self):
        return self._d
    
    #%%
    def get_basis(self):
        return self._basis
    
    #%%
    def get_params(self):
        return {'valid_outside': self._valid_outside,
                'zero_boundary': self._zero_boundary,
                'volume_perservation': self._volume_perservation,
                'Ashape': self._Ashape,
                'domain_min': self._domain_min,
                'domain_max': self._domain_max,
                'cell_increment': self._inc,
                'theta_dim': self._d,
                'original_dim': self._D}

    #%%
    def uniform_meshgrid(self, n_points):
        ''' '''
        if self._is_tf_tensor(n_points):
            lin_p = [tf.linspace(tf.cast(self._domain_min[i], tf.float32), 
                                 tf.cast(self._domain_max[i], tf.float32), n_points[i])
                    for i in range(self._ndim)]
            mesh_p = tf.meshgrid(*lin_p)                        
            grid = tf.concat([tf.reshape(array, (1, -1)) for array in mesh_p], axis=0)
            if not self._return_tf_tensors: grid = self._sess.run(grid)
        else:
            assert len(n_points) == self._ndim, \
                'n_points needs to be a list equal to the dimensionality of the transformation'
            lin_p = [np.linspace(self._domain_min[i], self._domain_max[i], n_points[i])
                    for i in range(self._ndim)]
            mesh_p = np.meshgrid(*lin_p)
            grid = np.vstack([array.flatten() for array in mesh_p])
            if self._return_tf_tensors: grid = tf.cast(grid, tf.float32)
        return grid
    
    #%%
    def sample_transformation(self, n_sample, mean=None, cov=None):
        ''' '''
        mean = np.zeros((self._d,)) if mean is None else mean
        cov = np.eye(self._d) if cov is None else cov
        theta = np.random.multivariate_normal(mean, cov, size=n_sample)
        if self._return_tf_tensors: theta = tf.cast(theta, tf.float32)
        return theta
    
    #%%
    def identity(self, n_sample):
        theta = np.zeros(shape=(n_sample, self._d))
        if self._return_tf_tensors: theta = tf.cast(theta, tf.float32)
        return theta

    #%%
    def transform_grid(self, points, theta):
        ''' '''
        # Assert if numpy array, else trust the user
        if not self._is_tf_tensor(points):
            assert points.shape[0] == self._ndim, \
             'Expects a grid of ' + str(self._ndim) + 'd points'            
        if not self._is_tf_tensor(theta):
            assert theta.shape[1] == self._d, \
                 'Expects theta to have shape N x ' + str(self._d)
        
        # Call transformer
        if self._return_tf_tensors:
            points = tf.cast(points, tf.float32)
            theta = tf.cast(theta, tf.float32)
            newpoints = self._transformer(points, theta)
        else:
            if self._fixed_data:
                newpoints = self._transformer_np(points, theta)    
            else:
                newpoints = self._sess.run(self._transformer(points, theta))
        return newpoints

    #%%
    def interpolate(self, data, transformed_points):
        ''' '''
        # Call interpolator 
        if self._return_tf_tensors:
            data = tf.cast(data, tf.float32)
            transformed_points = tf.cast(transformed_points, tf.float32)
            new_data = self._interpolate(data, transformed_points)
        else:
            if self._fixed_data:
                new_data = self._interpolate_np(data, transformed_points)
            else:
                new_data = self._sess.run(self._interpolate(data, transformed_points))
        return new_data
    
    #%%
    def transform_data(self, data, theta):
        ''' '''
        # Find grid size
        if self._is_tf_tensor(data):
            data_size = data.get_shape().as_list()[1:self._ndim+1]
        else:
            data_size = data.shape[1:self._ndim+1]
        
        # Fix equal size interpolation for 2D
        if self._ndim==2:
            data_size = data_size[::-1]
        
        # Create grid, call transformer, and interpolate
        points = self.uniform_meshgrid(data_size)
        new_points = self.transform_grid(points, theta)
        new_data = self.interpolate(data, new_points)
        return new_data
    
    #%%
    def fix_data_size(self, data_size):
        assert not self._return_tf_tensors, \
            " Cannot fix data size with return_tf_tensors true "
        data_p = tf.placeholder(tf.float32, (None, *data_size))
        theta_p = tf.placeholder(tf.float32, (None, self._d))
        if self._ndim==1 or self._ndim==3:
            points1_p = tf.placeholder(tf.float32, (self._ndim, np.prod(data_size)))
            points2_p = tf.placeholder(tf.float32, (None, self._ndim, np.prod(data_size)))
        else:
            points1_p = tf.placeholder(tf.float32, (self._ndim, np.prod(data_size[:2])))
            points2_p = tf.placeholder(tf.float32, (None, self._ndim, np.prod(data_size[:2])))
    
        self._transformer_np = self._sess.make_callable(self._transformer(points1_p, theta_p), 
                                                        [points1_p, theta_p])
        self._interpolate_np = self._sess.make_callable(self._interpolate(data_p, points2_p), 
                                                        [data_p, points2_p])
        
    #%%
    def calc_vectorfield(self, points, theta):
        # Construct the affine transformations
        Avees = self._theta2Avees(theta)
        As = self._Avees2As(Avees)
        
        # Find cells and extract correct affine transformation
        idx = self._sess.run(self._findcellidx(points.T, *self._nc))
        Aidx = As[idx]
        
        # Make homogeneous coordinates
        points = np.expand_dims(np.vstack((points, np.ones((1, points.shape[1])))).T,2)
        
        # Do matrix-vector multiplication
        v = np.matmul(Aidx, points)
        return np.squeeze(v).T
    
    #%%
    def visualize_vectorfield(self, theta, nb_points=10):
        points = self.uniform_meshgrid([nb_points for i in range(self._ndim)])
        v = self.calc_vectorfield(points, theta)
        
        # Plot
        import matplotlib.pyplot as plt
        if self._ndim==1:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.quiver(points[0,:], np.zeros_like(points), v, np.zeros_like(v))
            ax.set_xlim(self._domain_min[0], self._domain_max[0])
        elif self._ndim==2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.quiver(points[0,:], points[1,:], v[0,:], v[1,:])
            ax.set_xlim(self._domain_min[0], self._domain_max[0])
            ax.set_ylim(self._domain_min[1], self._domain_max[1])
        elif self._ndim==3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.quiver(points[0,:], points[1,:], points[2,:], v[0,:], v[1,:], v[2,:],
                      length=0.3, arrow_length_ratio=0.5)
            ax.set_xlim3d(self._domain_min[0], self._domain_max[0])
            ax.set_ylim3d(self._domain_min[1], self._domain_max[1])
            ax.set_zlim3d(self._domain_min[2], self._domain_max[2])
        plt.axis('equal')
        plt.title('Velocity field')
        plt.show()

    #%%
    def _theta2Avees(self, theta):
        Avees = self._basis.dot(theta)
        return Avees
    
    #%%
    def _Avees2As(self, Avees):
        As = np.reshape(Avees, (self._nC, *self._Ashape))
        return As
    
    #%%
    def _As2squareAs(self, As):
        squareAs = np.zeros(shape=(self._nC, self._ndim+1, self._ndim+1))
        squareAs[:,:-1,:] = As
        return squareAs
    
    #%%
    def _is_tf_tensor(self, tensor):
        return  isinstance(tensor, tf.Tensor) or \
                isinstance(tensor, tf.Variable)
        