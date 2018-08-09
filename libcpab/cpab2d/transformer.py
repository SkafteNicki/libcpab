# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:28:54 2018

@author: nsde
"""

#%%
import tensorflow as tf
from tensorflow.python.framework import function
from ..helper.tf_funcs import tf_repeat_matrix
from ..helper.tf_findcellidx import tf_findcellidx_2D
from ..helper.tf_expm import tf_expm3x3
from ..helper.utility import get_dir
from ..helper.utility import gpu_support as _gpu_support
from sys import platform as _platform

#%% Load dynamic module
def load_dynamic_modules():
    dir_path = get_dir(__file__)
    transformer_module = tf.load_op_library(dir_path + '/./CPAB_ops.so')
    transformer_op = transformer_module.calc_trans2
    grad_op = transformer_module.calc_grad2
    return transformer_op, grad_op

if _platform == "linux" or _platform == "linux2" or _platform == "darwin":    
    transformer_op, grad_op = load_dynamic_modules()

#%%
def tf_cpab_transformer_2D_pure(points, theta, tess):
    """ """
    with tf.name_scope('CPAB_transformer'):
        # Make sure that both inputs are in float32 format
        points = tf.cast(points, tf.float32) # format [2, nb_points]
        theta = tf.cast(theta, tf.float32) # format [n_theta, dim]
        n_theta = tf.shape(theta)[0]
        n_points = tf.shape(points)[1]
        
        # Repeat point matrix, one for each theta
        newpoints = tf_repeat_matrix(points, n_theta) # [n_theta, 2, nb_points]
        
        # Reshape into a [nb_points*n_theta, 2] matrix
        newpoints = tf.reshape(tf.transpose(newpoints, perm=[0,2,1]), (-1, tess.ndim))
        
        # Add a row of ones, creating a [nb_points*n_theta, 3] matrix
        newpoints = tf.concat([newpoints, tf.ones((n_theta*n_points, 1))], axis=1)
        
        # Expand dims for matrix multiplication later -> [nb_points*n_theta, 3, 1] tensor
        newpoints = tf.expand_dims(newpoints, axis=2)
        
        # Tessalation information
        nC = tf.cast(tess.nC, tf.int32)
        ncx = tf.cast(tess.nc[0], tf.int32)
        ncy = tf.cast(tess.nc[1], tf.int32)
        inc_x = tf.cast(tess.inc[0], tf.float32)
        inc_y = tf.cast(tess.inc[1], tf.float32)
        
        # Steps sizes
        nStepSolver = tess.nstepsolver
        dT = 1.0 / tf.cast(nStepSolver, tf.float32)
        
        # Get cpab basis
        B = tf.cast(tess.basis, tf.float32)

        # Repeat basis for batch multiplication
        B = tf_repeat_matrix(B, n_theta)
        
        # Calculate the row-flatted affine transformations Avees 
        Avees = tf.matmul(B, tf.expand_dims(theta, 2))
		
        # Reshape into (n_theta*number_of_cells, 2, 3) tensor
        As = tf.reshape(Avees, shape = (n_theta * nC, *tess.Ashape)) # format [n_theta * nC, 2, 3]
        
        # Multiply by the step size and do matrix exponential on each matrix
        Trels = tf_expm3x3(dT*As)
        Trels = tf.concat([Trels, tf.cast(tf.reshape(tf.tile([*(tess.ndim*[0]),1], 
                [n_theta*nC]), (n_theta*nC, 1, tess.ndim+1)), tf.float32)], axis=1)
        
        # Batch index to add to correct for the batch effect
        batch_idx = nC * tf.reshape(tf.transpose(tf.ones((n_points, n_theta), 
                    dtype=tf.int32)*tf.cast(tf.range(n_theta), tf.int32)),(-1,))
        
        # Body function for while loop (executes the computation)
        def body(i, points):
            # Find cell index of each point
            idx = tf_findcellidx_2D(points, ncx, ncy, inc_x, inc_y)
            
            # Correct for batch
            corrected_idx = tf.cast(idx, tf.int32) + batch_idx
            
            # Gether relevant matrices
            Tidx = tf.gather(Trels, corrected_idx)
            
            # Transform points
            newpoints = tf.matmul(Tidx, points)
            
            # Shape information is lost, but tf.while_loop requires shape 
            # invariance so we need to manually set it (easy in this case)
            newpoints.set_shape((None, tess.ndim+1, 1)) 
            return i+1, newpoints
        
        # Condition function for while loop (indicates when to stop)
        def cond(i, points):
            # Return iteration bound
            return tf.less(i, nStepSolver)
        
        # Run loop
        trans_points = tf.while_loop(cond, body, [tf.constant(0), newpoints],
                                     parallel_iterations=10, back_prop=True)[1]
        # Reshape to batch format
        trans_points = tf.reshape(tf.transpose(trans_points[:,:tess.ndim], perm=[1,0,2]), 
                                 (n_theta, tess.ndim, n_points))
        return trans_points

#%%
def _calc_trans(points, theta, tess):
    """ """
    with tf.name_scope('calc_trans'):
        # Make sure that both inputs are in float32 format
        points = tf.cast(points, tf.float32) # format [2, nb_points]
        theta = tf.cast(theta, tf.float32) # format [n_theta, dim]
        n_theta = tf.shape(theta)[0]
        
        # Tessalation information
        nC = tf.cast(tess.nC, tf.int32)
        ncx = tf.cast(tess.nc[0], tf.int32)
        ncy = tf.cast(tess.nc[1], tf.int32)
        inc_x = tf.cast(tess.inc[0], tf.float32)
        inc_y = tf.cast(tess.inc[1], tf.float32)
        
        # Steps sizes
        nStepSolver = tf.cast(tess.nstepsolver, dtype = tf.int32) 
        dT = 1.0 / tf.cast(nStepSolver , tf.float32)
        
        # Get cpab basis
        B = tf.cast(tess.basis, tf.float32)

        # Repeat basis for batch multiplication
        B = tf_repeat_matrix(B, n_theta)

        # Calculate the row-flatted affine transformations Avees 
        Avees = tf.matmul(B, tf.expand_dims(theta, 2))
		
        # Reshape into (number of cells, 2, 3) tensor
        As = tf.reshape(Avees, shape = (n_theta * nC, *tess.Ashape[0])) # format [n_theta * nC, 2, 3]
        
        # Multiply by the step size and do matrix exponential on each matrix
        Trels = tf_expm3x3(dT*As)
        Trels = Trels[:,:tess.ndim,:] # extract important part
        Trels = tf.reshape(Trels, shape=(n_theta, nC, *tess.Ashape[0]))
        
        # Call the dynamic library
        with tf.name_scope('calc_trans_op'):
	        newpoints = transformer_op(points, Trels, nStepSolver, ncx, ncy, inc_x, inc_y)
        return newpoints

#%%
def _calc_grad(op, grad):
    """ """
    with tf.name_scope('calc_grad'):
        # Grap input
        points = op.inputs[0] # 2 x nP
        theta = op.inputs[1] # n_theta x d
        tess = op.inputs[2] # tessalation
        n_theta = tf.shape(theta)[0]
        
        # Tessalation information
        nC = tf.cast(tess.nC, tf.int32)
        ncx = tf.cast(tess.nc[0], tf.int32)
        ncy = tf.cast(tess.nc[1], tf.int32)
        inc_x = tf.cast(tess.inc[0], tf.float32)
        inc_y = tf.cast(tess.inc[1], tf.float32)
        
        # Steps sizes
        nStepSolver = tf.cast(tess.nstepsolver, dtype = tf.int32)
    
        # Get cpab basis
        B = tf.cast(tess.basis, tf.float32)
        Bs = tf.reshape(tf.transpose(B), (-1, nC, *tess.Ashape))
        B = tf_repeat_matrix(B, n_theta)
        
        # Calculate the row-flatted affine transformations Avees 
        Avees = tf.matmul(B, tf.expand_dims(theta, 2))
        
        # Reshape into (ntheta, number of cells, 2, 3) tensor
        As = tf.reshape(Avees, shape = (n_theta, nC, *tess.Ashape)) # n_theta x nC x 2 x 3
        
        # Call cuda code
        with tf.name_scope('calc_grad_op'):
            # gradient: d x n_theta x 2 x n
            gradient = grad_op(points, As, Bs, nStepSolver, ncx, ncy, inc_x, inc_y) 
        
        # Reduce into: d x 1 vector
        gradient = tf.reduce_sum(grad * gradient, axis = [2,3])
        gradient = tf.transpose(gradient)
                                  
        return [None, gradient, None]

#%% Wrapper to connect function to gradient
@function.Defun(tf.float32, tf.float32, tf.variant, func_name='tf_CPAB_transformer_2D', python_grad_func=_calc_grad)
def tf_cpab_transformer_2D_cuda(points, theta, tess):
    return _calc_trans(points, theta, tess)

#%% Find out which version to use
_gpu = _gpu_support()
if _gpu:
    tf_cpab_transformer_2D = tf_cpab_transformer_2D_cuda
else:
    tf_cpab_transformer_2D = tf_cpab_transformer_2D_pure