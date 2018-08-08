#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:46:23 2017

@author: nsde
"""

#%%
import tensorflow as tf

#%%
def tf_mymin(x, y):
    """ Special min function for the findcellidx function """
    with tf.name_scope('mymin'):
        return tf.where(tf.less(x,y), x, tf.round(y))

#%%
def tf_findcellidx_1D(points, ncx, inc_x):
    """
    
    """
    with tf.name_scope('findcellidx_1D') :
        p = tf.squeeze(points)
        ncx, inc_x = tf.cast(ncx, tf.float32), tf.cast(inc_x, tf.float32)
        
        # Scale to [0,1] domain
        p = tf.cast(p[:,0] + 1, tf.float32) / 2.0
        
        # Floor values to find cell
        idx = tf.floor(p / (inc_x/2.0))

        idx = tf.clip_by_value(idx, clip_value_min=0, clip_value_max=ncx)
        idx = tf.cast(idx, tf.int32)
        return idx

#%%
def tf_findcellidx_2D(points, ncx, ncy, inc_x, inc_y):
    """ Computes the cell index for some points and a given tessalation 
    
    Arguments:
        points: 3D-`Tensor` [n_points,3,1], with points in homogeneous coordinates
        ncx, ncy: `integer`, with the number of cells in the x and y direction
        inc_x, inc_y: `floats`, the size of the cells in the x and y direction
    
    Output:
        idx: 1D-`Tensor` [n_points,], with the cell idx for each input point
    """
    with tf.name_scope('findcellidx_2D'):
        p = tf.transpose(tf.squeeze(points)) # 2 x n_points
        ncx, ncy = tf.cast(ncx, tf.float32), tf.cast(ncy, tf.float32)
        inc_x, inc_y = tf.cast(inc_x, tf.float32), tf.cast(inc_y, tf.float32)
    
        # Move according to lower bounds
        p = tf.cast(p + 1, tf.float32)
        
        p0 = tf.minimum((ncx*inc_x - 1e-8), tf.maximum(0.0, p[0,:]))
        p1 = tf.minimum((ncy*inc_y - 1e-8), tf.maximum(0.0, p[1,:]))
            
        xmod = tf.mod(p0, inc_x)
        ymod = tf.mod(p1, inc_y)
            
        x = xmod / inc_x
        y = ymod / inc_y
        
        # Calculate initial cell index    
        cell_idx =  tf_mymin((ncx - 1) * tf.ones_like(p0), (p0 - xmod) / inc_x) + \
                    tf_mymin((ncy - 1) * tf.ones_like(p0), (p1 - ymod) / inc_y) * ncx 
        cell_idx *= 4
    
        cell_idx1 = cell_idx+1
        cell_idx2 = cell_idx+2
        cell_idx3 = cell_idx+3

        # Conditions to evaluate        
        cond1 = tf.less_equal(p[0,:], 0) #point[0]<=0
        cond1_1 = tf.logical_and(tf.less_equal(p[1,:], 0), tf.less(p[1,:]/inc_y, 
            p[0,:]/inc_x))#point[1] <= 0 && point[1]/inc_y<point[0]/inc_x
        cond1_2 = tf.logical_and(tf.greater_equal(p[1,:], ncy*inc_y), tf.greater(
            p[1,:]/inc_y - ncy, -p[0,:]/inc_x))#(point[1] >= ncy*inc_y && point[1]/inc_y - ncy > point[0]/inc_x-ncx
        cond2 = tf.greater_equal(p[0,:], ncx*inc_x) #point[0] >= ncx*inc_x
        cond2_1 = tf.logical_and(tf.less_equal(p[1,:],0), tf.greater(-p[1,:]/inc_y,
            p[0,:]/inc_x-ncx))#point[1]<=0 && -point[1]/inc_y > point[0]/inc_x - ncx
        cond2_2 = tf.logical_and(tf.greater_equal(p[1,:],ncy*inc_y), tf.greater(
            p[1,:]/inc_y - ncy,p[0,:]/inc_x-ncx))#point[1] >= ncy*inc_y && point[1]/inc_y - ncy > point[0]/inc_x-ncx
        cond3 = tf.less_equal(p[1,:], 0) #point[1] <= 0
        cond4 = tf.greater_equal(p[1,:], ncy*inc_y) #point[1] >= ncy*inc_y
        cond5 = tf.less(x, y) #x<y
        cond5_1 = tf.less(1-x, y) #1-x<y
    
        # Take decision based on the conditions
        idx = tf.where(cond1, tf.where(cond1_1, cell_idx, tf.where(cond1_2, cell_idx2, cell_idx3)),
              tf.where(cond2, tf.where(cond2_1, cell_idx, tf.where(cond2_2, cell_idx2, cell_idx1)),
              tf.where(cond3, cell_idx, 
              tf.where(cond4, cell_idx2,
              tf.where(cond5, tf.where(cond5_1, cell_idx2, cell_idx3), 
              tf.where(cond5_1, cell_idx1, cell_idx))))))
    
        return idx

#%%
def tf_repeat_matrix(x_in, n_repeats):
    """ Concatenate n_repeats copyies of x_in by a new 0 axis """
    with tf.name_scope('repeat_matrix'):
        x_out = tf.reshape(x_in, (-1,))
        x_out = tf.tile(x_out, [n_repeats])
        x_out = tf.reshape(x_out, (n_repeats, tf.shape(x_in)[0], tf.shape(x_in)[1]))
        return x_out

#%%
def tf_repeat(x, n_repeats):
    """
    Tensorflow implementation of np.repeat(x, n_repeats)
    """
    with tf.name_scope('repeat'):
        ones = tf.ones(shape=(n_repeats, ))
        rep = tf.transpose(tf.expand_dims(ones, 1), [1, 0])
        rep = tf.cast(rep, x.dtype)
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])