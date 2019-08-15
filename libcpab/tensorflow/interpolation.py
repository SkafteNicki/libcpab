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
    
    # Problem size
    n_batch = data.shape[0]
    width = data.shape[1]
    height = data.shape[2]
    n_channels = data.shape[3]
    out_width, out_height = outsize
    
    # Cast value to float dtype
    x = tf.reshape(grid[:,0], (-1, ))
    y = tf.reshape(grid[:,1], (-1, ))
    max_x = tf.cast(width - 1, tf.int32)
    max_y = tf.cast(height - 1, tf.int32)

    # Scale indices from [0, 1] to [0, width/height]
    x = x * (width-1)
    y = y * (height-1)

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
    
    # Batch effect
    batch_size = out_width*out_height
    batch_idx = tf.tile(tf.range(n_batch), (batch_size,))
    
    # Index
    c00 = tf.gather_nd(data, tf.stack([batch_idx, x0, y0], axis=1))
    c01 = tf.gather_nd(data, tf.stack([batch_idx, x0, y1], axis=1))
    c10 = tf.gather_nd(data, tf.stack([batch_idx, x1, y0], axis=1))
    c11 = tf.gather_nd(data, tf.stack([batch_idx, x1, y1], axis=1))

    # Interpolation weights
    xd = tf.reshape(x-tf.cast(x0, tf.float32), (-1,1))
    yd = tf.reshape(y-tf.cast(y0, tf.float32), (-1,1))
    
    # Do interpolation
    c0 = c00*(1-xd) + c10*xd
    c1 = c01*(1-xd) + c11*xd
    c = c0*(1-yd) + c1*yd
    
    # Reshape
    newim = tf.reshape(c, (n_batch, out_height, out_width, n_channels))
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