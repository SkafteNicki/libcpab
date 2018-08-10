#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:37:16 2018

@author: nsde
"""

import tensorflow as tf

#%%
def tf_repeat(x, n_repeats):
    """ Tensorflow implementation of np.repeat(x, n_repeats) """
    with tf.name_scope('repeat'):
        ones = tf.ones(shape=(n_repeats, ))
        rep = tf.transpose(tf.expand_dims(ones, 1), [1, 0])
        rep = tf.cast(rep, x.dtype)
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

#%%
def tf_interpolate_1D(data, grid):
    " data: n_batch x length timeseries "
    " grid: n_batch x 1 x n_points "
    with tf.name_scope('interpolate'):
        data, grid = tf.cast(data, tf.float32), tf.cast(grid, tf.float32)
        
        # Constants
        n_batch = tf.shape(data)[0]
        _, length_d = data.shape.as_list()
        _, _, length_g = grid.shape.as_list()
        max_x = tf.cast(length_d-1, tf.int32)
        
        # Cast
        x = tf.cast(tf.reshape(grid[:,0], (-1,)), tf.float32)
        
        # Scale to domain
        x = x * length_d
        
        # Do sampling
        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        
        # Clip values
        x0 = tf.clip_by_value(x0, 0, max_x)
        x1 = tf.clip_by_value(x1, 0, max_x)
        
        # Take care of batch effect
        base = tf_repeat(tf.range(n_batch)*length_d, length_g)
        idx_1 = base + x0
        idx_2 = base + x1
        
        # Lookup values
        data_flat = tf.reshape(data, (-1,))
        i1 = tf.gather(data_flat, idx_1)
        i2 = tf.gather(data_flat, idx_2)
        
        # Convert to floats
        x0 = tf.cast(x0, tf.float32)
        x1 = tf.cast(x1, tf.float32)
        
        # Interpolation weights
        w1 = (x - x0)
        w2 = (x1 - x)
        
        # Do interpolation
        new_data = w1*i1 + w2*i2
        
        # Reshape and return
        new_data = tf.reshape(new_data, (n_batch, length_g))
        return new_data