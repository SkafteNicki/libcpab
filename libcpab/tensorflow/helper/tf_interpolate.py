#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 08:04:47 2018

@author: nsde
"""

#%%
import tensorflow as tf
from ..helper.tf_funcs import tf_repeat, tf_img_normalize, tf_shape_i

#%%
def tf_interpolate_1D(data, grid):
    " data: n_batch x length timeseries "
    " grid: n_batch x 1 x n_points "
    with tf.name_scope('interpolate'):
        data, grid = tf.cast(data, tf.float32), tf.cast(grid, tf.float32)
        
        # Constants
        n_batch = tf_shape_i(data,0)
        length_d = tf_shape_i(data,1)
        length_g = tf_shape_i(grid,2)
        max_x = tf.cast(length_d-1, tf.int32)
        
        # Extract points
        x = tf.reshape(grid[:,0], (-1,)) # [n_theta x n_points]
        
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
        
        # Do interpolation
        new_data = i1 + (x - x0) * (i2 - i1)#w1*i1 + w2*i2
        
        # Reshape and return
        new_data = tf.reshape(new_data, (n_batch, length_g))
        return new_data

#%%
def tf_interpolate_2D(data, grid):
    with tf.name_scope('interpolate'):
        data, grid = tf.cast(data, tf.float32), tf.cast(grid, tf.float32)
        
        # Normalize image values
        data = tf_img_normalize(data)
        
        # Constants
        n_batch = tf_shape_i(data,0)
        height = tf_shape_i(data,1)
        width = tf_shape_i(data,2)
        n_channels = tf_shape_i(data,3)

        # Cast value to float dtype
        x = tf.reshape(grid[:,0], (-1, ))
        y = tf.reshape(grid[:,1], (-1, ))
        height_f = tf.cast(height, tf.float32)
        width_f = tf.cast(width, tf.float32)
        max_x = tf.cast(width - 1, tf.int32)
        max_y = tf.cast(height - 1, tf.int32)

        # Scale indices from [0, 1] to [0, width/height]
        x = x * width_f
        y = y * height_f

        # Do sampling
        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        # Find index of each corner point
        x0 = tf.clip_by_value(x0, 0, max_x)
        x1 = tf.clip_by_value(x1, 0, max_x)
        y0 = tf.clip_by_value(y0, 0, max_y)
        y1 = tf.clip_by_value(y1, 0, max_y)
        dim1 = width*height
        dim2 = width
        base = tf_repeat(tf.range(n_batch)*dim1, dim1)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # Use indices to lookup pixels in the flat image and restore
        # channels dim
        data_flat = tf.reshape(data, (-1, n_channels))
        data_flat = tf.cast(data_flat, tf.float32)
        Ia = tf.gather(data_flat, idx_a)
        Ib = tf.gather(data_flat, idx_b)
        Ic = tf.gather(data_flat, idx_c)
        Id = tf.gather(data_flat, idx_d)

        # And finally calculate interpolated values
        x0_f = tf.cast(x0, tf.float32)
        x1_f = tf.cast(x1, tf.float32)
        y0_f = tf.cast(y0, tf.float32)
        y1_f = tf.cast(y1, tf.float32)
        wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
        wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
        wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
        wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
        newim = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
        
        # Reshape into image format and take care of numeric underflow/overflow
        newim = tf.reshape(newim, (n_batch, height, width, n_channels))
        newim = tf.clip_by_value(newim, 0.0, 1.0)
        return newim

#%%
def tf_interpolate_3D(data, grid):
    " data: n_batch x width x height x depth "
    " grid: n_batch x 3 x n_points "
    with tf.name_scope('interpolate'):
        data, grid = tf.cast(data, tf.float32), tf.cast(grid, tf.float32)
        
        # Constants
        n_batch = tf_shape_i(data,0)
        width = tf_shape_i(data,1)
        height = tf_shape_i(data,2)
        depth = tf_shape_i(data,3)
        max_x = tf.cast(width - 1, tf.int32)
        max_y = tf.cast(height - 1, tf.int32)
        max_z = tf.cast(depth - 1, tf.int32)
        
        # Extact points
        x = tf.reshape(grid[:,0], (-1,))
        y = tf.reshape(grid[:,1], (-1,))
        z = tf.reshape(grid[:,2], (-1,))
        
        # Scale to domain
        x = x * (width-1)
        y = y * (height-1)
        z = z * (depth-1)
        
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
        dim1 = depth * height * width
        dim2 = depth * height
        dim3 = depth
        base = tf_repeat(tf.range(n_batch) * dim1, dim1)
        base_z0 = base + dim2 * z0
        base_z1 = base + dim2 * z1
        base_z0_y0 = base_z0 + dim3 * y0
        base_z0_y1 = base_z0 + dim3 * y1
        base_z1_y0 = base_z1 + dim3 * y0
        base_z1_y1 = base_z1 + dim3 * y1
        i1 = base_z0_y0 + x0
        i2 = base_z0_y0 + x1
        i3 = base_z0_y1 + x0
        i4 = base_z0_y1 + x1
        i5 = base_z1_y0 + x0
        i6 = base_z1_y0 + x1
        i7 = base_z1_y1 + x0
        i8 = base_z1_y1 + x1

        # Lookup values
        data_flat = tf.reshape(data, (-1, ))
        c000 = tf.gather(data_flat, i1)
        c001 = tf.gather(data_flat, i5)
        c010 = tf.gather(data_flat, i2)
        c011 = tf.gather(data_flat, i6)
        c100 = tf.gather(data_flat, i3)
        c101 = tf.gather(data_flat, i7)
        c110 = tf.gather(data_flat, i4)
        c111 = tf.gather(data_flat, i8)
        
        # Float casting
        x0_f = tf.cast(x0, tf.float32)
        y0_f = tf.cast(y0, tf.float32)
        z0_f = tf.cast(z0, tf.float32)
        
        
        # Interpolation weights
        xd = (x-x0_f)
        yd = (y-y0_f)
        zd = (z-z0_f)
        
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
    
#%%
if __name__ == '__main__':
    pass