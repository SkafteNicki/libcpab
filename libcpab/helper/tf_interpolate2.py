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
        length_d = tf.shape(data)[1]
        length_g = tf.shape(data)[2]
        max_x = tf.cast(length_d-1, tf.int32)
        
        # Extract points
        x = tf.reshape(grid[:,0], (-1,))
        
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
    
#%%
def tf_interpolate_2D(data, grid):
    " data: n_batch x width x height x n_channel "
    " grid: n_batch x 2 x n_points "
    with tf.name_scope('interpolate'):
        data, grid = tf.cast(data, tf.float32), tf.cast(grid, tf.float32)
        
        # Constants
        n_batch = tf.shape(data)[0]
        height = tf.shape(data)[1]
        width = tf.shape(data)[2]
        n_channels = tf.shape(data)[3]
        max_x = tf.cast(width - 1, tf.int32)
        max_y = tf.cast(height - 1, tf.int32)
        
        # Extact points
        x = tf.reshape(grid[:,0], (-1,))
        y = tf.reshape(grid[:,1], (-1,))
        
        # Scale to domain
        x = x * tf.cast(width, tf.float32)
        y = y * tf.cast(height, tf.float32)
        
        # Do sampling
        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1
        
        # Clip values
        x0 = tf.clip_by_value(x0, 0, max_x)
        x1 = tf.clip_by_value(x1, 0, max_x)
        y0 = tf.clip_by_value(y0, 0, max_y)
        y1 = tf.clip_by_value(y1, 0, max_y)
        
        # Take care of batch effect
        dim1, dim2 = width*height, width
        base = tf_repeat(tf.range(n_batch)*dim1, dim1)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # Lookup values
        data_flat = tf.reshape(data, (-1, n_channels))
        Ia = tf.gather(data_flat, idx_a)
        Ib = tf.gather(data_flat, idx_b)
        Ic = tf.gather(data_flat, idx_c)
        Id = tf.gather(data_flat, idx_d)
        
        # Convert to floats
        x0 = tf.cast(x0, tf.float32)
        x1 = tf.cast(x1, tf.float32)
        y0 = tf.cast(y0, tf.float32)
        y1 = tf.cast(y1, tf.float32)
        
        # Interpolation weights
        wa = tf.expand_dims(((x1-x) * (y1-y)), 1)
        wb = tf.expand_dims(((x1-x) * (y-y0)), 1)
        wc = tf.expand_dims(((x-x0) * (y1-y)), 1)
        wd = tf.expand_dims(((x-x0) * (y-y0)), 1)
        
        # Do interpolation
        new_data = wa*Ia + wb*Ib + wc*Ic + wd*Id
        
        # Reshape and return
        new_data = tf.clip_by_value(new_data, tf.reduce_min(data), tf.reduce_max(data))
        new_data = tf.reshape(new_data, (n_batch, height, width, n_channels))
        return new_data
    
#%%
def tf_interpolate_3D(data, grid):
    " data: n_batch x width x height x depth "
    " grid: n_batch x 3 x n_points "
    with tf.name_scope('interpolate'):
        data, grid = tf.cast(data, tf.float32), tf.cast(grid, tf.float32)
        
        # Constants
        n_batch = tf.shape(data)[0]
        width = tf.shape(data)[1]
        height = tf.shape(data)[2]
        depth = tf.shape(data)[3]
        max_x = tf.cast(width - 1, tf.int32)
        max_y = tf.cast(height - 1, tf.int32)
        max_z = tf.cast(depth - 1, tf.int32)
        
        # Extact points
        x = tf.reshape(grid[:,0], (-1,))
        y = tf.reshape(grid[:,1], (-1,))
        z = tf.reshape(grid[:,2], (-1,))
        
        # Scale to domain
        x = x * width
        y = y * height
        z = z * depth
        
        # Do sampling
        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1
        z0 = tf.cast(tf.floor(z), tf.int32)
        z1 = z0 + 1
        
        # Clip values
        x0 = tf.clip_by_value(x0, 0, max_x)
        x1 = tf.clip_by_value(x1, 0, max_x)
        y0 = tf.clip_by_value(y0, 0, max_y)
        y1 = tf.clip_by_value(y1, 0, max_y)
        z0 = tf.clip_by_value(z0, 0, max_z)
        z1 = tf.clip_by_value(z1, 0, max_z)
        
        # Take care of batch effect
        base = tf_repeat(tf.range(n_batch), width * height * depth)

        # Lookup values
        c000 = tf.gather_nd(data, [base,x0,y0,z0])
        c001 = tf.gather_nd(data, [base,x0,y0,z1])
        c010 = tf.gather_nd(data, [base,x0,y1,z0])
        c011 = tf.gather_nd(data, [base,x0,y1,z1])
        c100 = tf.gather_nd(data, [base,x1,y0,z0])
        c101 = tf.gather_nd(data, [base,x1,y0,z1])
        c110 = tf.gather_nd(data, [base,x1,y1,z0])
        c111 = tf.gather_nd(data, [base,x1,y1,z1])
        
        # Interpolation weights
        xd = (x-x0)/(x1-x0)
        yd = (y-y0)/(y1-y0)
        zd = (z-z0)/(z1-z0)
        
        # Do interpolation
        c00 = c000*(1-xd) + c100*xd
        c01 = c001*(1-xd) + c101*xd
        c10 = c010*(1-xd) + c110*xd
        c11 = c011*(1-xd) + c111*xd
        c0 = c00*(1-yd) + c10*yd
        c1 = c01*(1-yd) + c11*yd
        c = c0*(1-zd) + c1*zd
        
        # Reshape and return
        new_data = tf.reshape(c, (n_batch, width, height, depth))
        return new_data

        
        