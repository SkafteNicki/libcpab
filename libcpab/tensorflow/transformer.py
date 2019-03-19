# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 10:27:16 2018

@author: nsde
"""

#%%
import tensorflow as tf
from .findcellidx import findcellidx
from .expm import expm
from ..helper.utility import get_dir
from sys import platform as _platform

#%%
# Small helper function that does nothing
def f(*args):
    return None

import pdb
pdb.set_trace()

#%%
_verbose = False
try:
    dir_path = get_dir(__file__)
    transformer_module = tf.load_op_library(dir_path + '/./transformer.so')
    transformer_op = transformer_module.calc_trans
    grad_op = transformer_module.calc_grad
    compiled = True
except Exception as e:
    transformer_op = f
    grad_op = f
    compiled = False
    if _verbose:
        print(70*'=')
        print('Unsuccesfully compiled gpu source')
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
    # Problem parameters
    n_theta = theta.shape[0]
    n_points = points.shape[1]
    
    # Create homogenous coordinates
    ones = tf.ones((n_theta, 1, n_points)).to(points.device)
    if len(points) == 2:
        newpoints = points[None].repeat(n_theta, 1, 1) # [n_theta, ndim, n_points]
    newpoints = tf.concat((newpoints, ones), axis=1) # [n_theta, ndim+1, n_points]
    newpoints = newpoints.permute(0, 2, 1) # [n_theta, n_points, ndim+1]
    newpoints = tf.reshape(newpoints, (-1, params.ndim+1)) #[n_theta*n_points, ndim+1]]
    newpoints = newpoints[:,:,None] # [n_theta*n_points, ndim+1, 1]

    # Get velocity fields
    B = tf.cast(params.basis, dtype=tf.float32, device=theta.device)
    Avees = tf.matmul(B, tf.transpose(theta))
    As = tf.reshape(tf.transpose(Avees), (n_theta*params.nC, *params.Ashape))
    zero_row = tf.zeros((n_theta*params.nC, 1, params.ndim+1), device=As.device)
    AsSquare = tf.concat([As, zero_row], axis=1)

    # Take matrix exponential
    dT = 1.0 / params.nstepsolver
    Trels = expm(dT*AsSquare)
    
    # Take care of batch effect
    batch_idx = params.nC*(tf.ones((n_points, n_theta), dtype=tf.int64) * tf.range(n_theta))
    batch_idx = batch_idx.flatten().to(theta.device)
    
    # Do integration
    for i in range(params.nstepsolver):
        idx = findcellidx(params.ndim, tf.transpose(newpoints[:,:,0]), params.nc) + batch_idx
        Tidx = Trels[idx]
        newpoints = tf.matmul(Tidx, newpoints)
        
    newpoints = tf.transpose(newpoints.squeeze()[:,:params.ndim])
    newpoints = tf.transpose(tf.reshape(newpoints, (params.ndim, n_theta, n_points)), perm=[1,0,2])
    return newpoints

#%%
@tf.custom_gradient
def CPAB_transformer_fast(points, theta, params):
    device = theta.device
    n_theta = theta.shape[0]
    
    # Get Volocity fields
    B = tf.cast(params.basis, dtype=tf.float32, device=device)
    Avees = tf.matmul(B, tf.transpose(theta))
    As = tf.reshape(tf.transpose(Avees), (n_theta*params.nC, *params.Ashape))
    zero_row = tf.zeros((n_theta*params.nC, 1, params.ndim+1), device=device)
    AsSquare = tf.concat([As, zero_row], axis=1)
    
    # Take matrix exponential
    dT = 1.0 / params.nstepsolver
    Trels = expm(dT * AsSquare)
    Trels = tf.reshape(Trels[:,:params.ndim,:], (n_theta, params.nC, *params.Ashape))
    
    # Convert to tensor
    nstepsolver = tf.cast(params.nstepsolver, dtype=tf.int32, device=device)
    nc = tf.cast(params.nc, dtype=tf.int32, device=device)
    
    # Call integrator
    newpoints = transformer_op(points, Trels, nstepsolver, nc)
    Bs = tf.reshape(tf.transpose(B), (-1, params.nC, *params.Ashape))
    As = tf.reshape(As, (n_theta, params.nC, *params.Ashape))
    
    def grad(grad):
        gradient = grad_op(points, As, Bs, nstepsolver, nc)
        g = tf.reduce_sum(gradient * grad, axis=[2,3])
        return None, g, None
    
    return newpoints, grad