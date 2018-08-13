#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 08:04:47 2018

@author: nsde
"""

#%%
import tensorflow as tf
from .tf_funcs import tf_repeat, tf_img_normalize

#%%
def tf_interpolate_1D(x, x_trans, y, ts_length):
    with tf.name_scope('interpolate'):
        # Find nearest smaller neighbor
        dist = tf.subtract(tf.reshape(x_trans, (-1, 1)), x)
        
        # Find index of interval in tessellation
        greater_than_zero = tf.cast(tf.greater_equal(dist, 0), tf.float32)
        idx = (ts_length-1) - tf.reduce_sum(greater_than_zero, axis=0)
        idx = tf.clip_by_value(idx, clip_value_min=0, clip_value_max=ts_length-2)
        
        # Fetch values from x_trans and y
        x0 = tf.gather(x_trans, idx)
        x1 = tf.gather(x_trans, idx+1)
        y0 = tf.gather(y, idx)
        y1 = tf.gather(y, idx+1)    
        
        # Linear interpolation
        y_interp = y0 + (x-x0) * ((y1-y0)/(x1-x0))
        return y_interp

#%%
def tf_interpolate_2D(im, x, y, out_size):
    with tf.name_scope('interpolate'):
        # Normalize image values
        im = tf_img_normalize(im)
        
        # Constants
        n_batch = tf.shape(im)[0] # (often) unknown size
        #_, height, width, n_channels = im.shape.as_list() # known sizes
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        n_channels = tf.shape(im)[3]

        # Cast value to float dtype
        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        zero = tf.zeros([], dtype='int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

        # Scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0)*(width_f) / 2.0
        y = (y + 1.0)*(height_f) / 2.0

        # Do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        # Find index of each corner point
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width*height
        base = tf_repeat(tf.range(n_batch)*dim1, out_height*out_width)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # Use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, (-1, n_channels))
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # And finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
        wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
        wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
        wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
        newim = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
        
        # Reshape into image format and take care of numeric underflow/overflow
        newim = tf.reshape(newim, (n_batch, out_height, out_width, n_channels))
        newim = tf.clip_by_value(newim, 0.0, 1.0)
        return newim

#%%
def tf_interpolate_3D(volume, trn):
    with tf.name_scope('interpolate'):
        w, d, h = tf.shape(volume)[0] , tf.shape(volume)[1], tf.shape(volume)[2]
        trn = tf.minimum(tf.maximum(trn, 1e-5), 1-1e-5)
        
        x = tf.cast((w-1)*trn[0], tf.float32)
        y = tf.cast((d-1)*trn[1], tf.float32)
        z = tf.cast((h-1)*trn[2], tf.float32)
    
        x0 = tf.floor(x)
        x1 = x0+1
        y0 = tf.floor(y)
        y1 = y0+1
        z0 = tf.floor(z)
        z1 = z0+1
    
        xd = (x-x0)/(x1-x0)
        yd = (y-y0)/(y1-y0)
        zd = (z-z0)/(z1-z0)
    
        c000 = tf.gather_nd(volume, [x0,y0,z0])
        c001 = tf.gather_nd(volume, [x0,y0,z1])
        c010 = tf.gather_nd(volume, [x0,y1,z0])
        c011 = tf.gather_nd(volume, [x0,y1,z1])
        c100 = tf.gather_nd(volume, [x1,y0,z0])
        c101 = tf.gather_nd(volume, [x1,y0,z1])
        c110 = tf.gather_nd(volume, [x1,y1,z0])
        c111 = tf.gather_nd(volume, [x1,y1,z1])
    
        c00 = c000*(1-xd) + c100*xd
        c01 = c001*(1-xd) + c101*xd
        c10 = c010*(1-xd) + c110*xd
        c11 = c011*(1-xd) + c111*xd
    
        c0 = c00*(1-yd) + c10*yd
        c1 = c01*(1-yd) + c11*yd
    
        c = c0*(1-zd) + c1*zd
    
        return c