# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 10:27:16 2018

@author: nsde
"""

#%%
import tensorflow as tf
from .findcellidx import findcellidx
from .expm import expm
from ..core.utility import get_dir
from sys import platform as _platform

#%%
# Small helper function that does nothing
def f(*args):
    return None

#%%
_verbose = True
try:
    dir_path = get_dir(__file__)
    transformer_module = tf.load_op_library(dir_path + '/trans_op.so')
    transformer_op = transformer_module.calc_trans
    grad_op = transformer_module.calc_grad
    compiled = True
    if _verbose:
        print(70*'=')
        print('Succesfully loaded c++ source')
except Exception as e:
    transformer_op = f
    grad_op = f
    compiled = False
    if _verbose:
        print(70*'=')
        print('Unsuccesfully loaded c++ source')
        print('Error was: ')
        print(e)

#%%
def CPAB_transformer(points, theta, params):
    if not params.use_slow and compiled:
        return CPAB_transformer_fast(points, theta, params)
    else:
        return CPAB_transformer_slow(points, theta, params)
    
#%%
def CPAB_transformer_slow(points, theta, params):
    with tf.device(points.device):
        # Problem parameters
        n_theta = theta.shape[0]
        n_points = points.shape[1]
    
        # Create homogenous coordinates
        ones = tf.ones((n_theta, 1, n_points))
        if len(points) == 2:
            newpoints = tf.tile(points[None], (n_theta, 1, 1)) # [n_theta, ndim, n_points]
        newpoints = tf.concat((newpoints, ones), axis=1) # [n_theta, ndim+1, n_points]
        newpoints = tf.transpose(newpoints, perm=(0, 2, 1)) # [n_theta, n_points, ndim+1]
        newpoints = tf.reshape(newpoints, (-1, params.ndim+1)) #[n_theta*n_points, ndim+1]]
        newpoints = newpoints[:,:,None] # [n_theta*n_points, ndim+1, 1]

        # Get velocity fields
        B = tf.cast(params.basis, dtype=tf.float32)
        zero_row = tf.zeros((n_theta*params.nC, 1, params.ndim+1))
        Avees = tf.matmul(B, tf.transpose(theta))
        As = tf.reshape(tf.transpose(Avees), (n_theta*params.nC, *params.Ashape))
        AsSquare = tf.concat([As, zero_row], axis=1)

        # Take matrix exponential
        dT = 1.0 / params.nstepsolver
        Trels = expm(dT*AsSquare)
    
        # Take care of batch effect
        batch_idx = params.nC*(tf.ones((n_points, n_theta), dtype=tf.int32) * tf.range(n_theta))
        batch_idx = tf.reshape(tf.transpose(batch_idx), (-1,))
    
        # Do integration
        for i in range(params.nstepsolver):
            idx = findcellidx(params.ndim, tf.transpose(newpoints[:,:,0]), params.nc) + batch_idx
            Tidx = tf.gather(Trels, idx)
            newpoints = tf.matmul(Tidx, newpoints)
    
        newpoints = tf.transpose(tf.squeeze(newpoints)[:,:params.ndim])
        newpoints = tf.transpose(tf.reshape(newpoints, (params.ndim, n_theta, n_points)), perm=[1,0,2])
        return newpoints

#%%
def CPAB_transformer_fast(points, theta, params):
    # This is just stupid. We need the tf.custom_gradient decorator to equip the fast
    # gpu version with the gradient operation. But by using the decorator, we can only
    # call the actual function with tensor-input, and params is a dict. Therefore, this
    # small work around
    @tf.custom_gradient
    def actual_function(points, theta):
        device = theta.device
        n_theta = theta.shape[0]
        
        # Get Volocity fields
        with tf.device(device):
            B = tf.cast(params.basis, dtype=tf.float32)
        Avees = tf.matmul(B, tf.transpose(theta))
        As = tf.reshape(tf.transpose(Avees), (n_theta*params.nC, *params.Ashape))
        with tf.device(device):
            zero_row = tf.zeros((n_theta*params.nC, 1, params.ndim+1))
        AsSquare = tf.concat([As, zero_row], axis=1)
        
        # Take matrix exponential
        dT = 1.0 / params.nstepsolver
        Trels = expm(dT * AsSquare)
        Trels = tf.reshape(Trels[:,:params.ndim,:], (n_theta, params.nC, *params.Ashape))
        
        # Convert to tensor
        with tf.device(device):
            nstepsolver = tf.cast(params.nstepsolver, dtype=tf.int32)
            nc = tf.cast(params.nc, dtype=tf.int32)
        
        # Call integrator
        newpoints = transformer_op(points, Trels, nstepsolver, nc)
        Bs = tf.reshape(tf.transpose(B), (-1, params.nC, *params.Ashape))
        As = tf.reshape(As, (n_theta, params.nC, *params.Ashape))
        
        def grad(grad):
            gradient = grad_op(points, As, Bs, nstepsolver, nc)
            g = tf.reduce_sum(gradient * grad, axis=[2,3])
            return None, g, None
    
        return newpoints, grad
    return actual_function(points, theta)
