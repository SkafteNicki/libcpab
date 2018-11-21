# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 10:26:30 2018

@author: nsde
"""

#%%
import tensorflow as tf
from .helper import tf_shape_i, tf_repeat

#%%
def interpolate(ndim, data, grid, outsize):
    if ndim==1: return interpolate1D(data, grid, outsize)
    elif ndim==2: return interpolate2D(data, grid, outsize)
    elif ndim==3: return interpolate3D(data, grid, outsize)

#%%    
def interpolate1D(data, grid, outsize):
    data, grid = tf.cast(data, tf.float32), tf.cast(grid, tf.float32)
        
    # Constants
    n_batch = tf_shape_i(data,0)
    length_d = tf_shape_i(data,1)
    n_channel = tf_shape_i(data,2)
    length_g = tf_shape_i(grid,2)
    max_x = tf.cast(length_d-1, tf.int32)
        
    # Extract points
    x = tf.reshape(grid[:,0], (-1,)) # [n_theta x n_points]
    
    # Scale to domain
    x = x * (length_d-1)
        
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
    data_flat = tf.reshape(data, (-1,n_channel))
    i1 = tf.gather(data_flat, idx_1)
    i2 = tf.gather(data_flat, idx_2)
        
    # Convert to floats
    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
        
    # Do interpolation
    new_data = i1 + tf.transpose((x - x0) * tf.transpose((i2 - i1), perm=[1,0]), perm=[1,0]) #w1*i1 + w2*i2
        
    # Reshape and return
    new_data = tf.reshape(new_data, (n_batch, length_g, n_channel))
    return new_data

#%%    
def interpolate2D(data, grid, outsize):
    data, grid = tf.cast(data, tf.float32), tf.cast(grid, tf.float32)
    
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
def interpolate3D(data, grid, outsize):
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
    
    dim1 = width * height * depth
    base = tf_repeat(tf.range(n_batch) * dim1, dim1)
    
    c000 = tf.gather_nd(data, tf.stack([base, x0, y0, z0], axis=1))
    c001 = tf.gather_nd(data, tf.stack([base, x0, y0, z1], axis=1))
    c010 = tf.gather_nd(data, tf.stack([base, x0, y1, z0], axis=1))
    c011 = tf.gather_nd(data, tf.stack([base, x0, y1, z1], axis=1))
    c100 = tf.gather_nd(data, tf.stack([base, x1, y0, z0], axis=1))
    c101 = tf.gather_nd(data, tf.stack([base, x1, y0, z1], axis=1))
    c110 = tf.gather_nd(data, tf.stack([base, x1, y1, z0], axis=1))
    c111 = tf.gather_nd(data, tf.stack([base, x1, y1, z1], axis=1))
    
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
    new_data = tf.transpose(tf.reshape(c, (n_batch, depth, height, width)), perm=[0,3,2,1])
    return new_data