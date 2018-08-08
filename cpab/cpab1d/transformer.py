# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:28:54 2018

@author: nsde
"""

#%%
import tensorflow as tf
from ..helper.tf_funcs import tf_repeat_matrix

#%%
def tf_cpab_transformer_1D(points, theta, tess):
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
        newpoints = tf.reshape(tf.transpose(newpoints, perm=[0,2,1]), (-1, 2))
        
        # Add a row of ones, creating a [nb_points*n_theta, 3] matrix
        newpoints = tf.concat([newpoints, tf.ones((n_theta*n_points, 1))], axis=1)
        
        # Expand dims for matrix multiplication later -> [nb_points*n_theta, 3, 1] tensor
        newpoints = tf.expand_dims(newpoints, 2)
        
        # Tessalation information
        ncx = tf.cast(tess.nc[0], tf.int32)
        inc_x = tf.cast(tess.inc[0], tf.float32)