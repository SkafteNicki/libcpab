#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 09:55:27 2018

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
    Arguments:
        points: [n_points, 2, 1]
    """
    with tf.name_scope('findcellidx_1D') :
        p = points[:,0]
        ncx, inc_x = tf.cast(ncx, tf.float32), tf.cast(inc_x, tf.float32)
                
        # Floor values to find cell
        idx = tf.floor(p / inc_x)

        idx = tf.clip_by_value(idx, clip_value_min=0, clip_value_max=ncx)
        idx = tf.cast(idx, tf.int32)
        return idx

#%%
def tf_findcellidx_2D(points, ncx, ncy, inc_x, inc_y):
    """ """
    with tf.name_scope('findcellidx_2D'):
        p = tf.cast(tf.transpose(tf.squeeze(points)), tf.float32) # 3 x n_points
        ncx, ncy = tf.cast(ncx, tf.float32), tf.cast(ncy, tf.float32)
        inc_x, inc_y = tf.cast(inc_x, tf.float32), tf.cast(inc_y, tf.float32)

        # Determine inner coordinates        
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
def tf_findcellidx_3D(points, ncx, ncy, ncz, inc_x, inc_y, inc_z):
    with tf.name_scope('findcellidx_3D'):
        p = tf.cast(tf.transpose(tf.squeeze(points)), tf.float32) # 4 x n_points
        ncx, ncy, ncz = tf.cast(ncx, tf.float32), tf.cast(ncy, tf.float32), tf.cast(ncz, tf.float32)
        inc_x, inc_y, inc_z = tf.cast(inc_x, tf.float32), tf.cast(inc_y, tf.float32), tf.cast(inc_z, tf.float32)
        
        # Initial row, col placement
        p0 = tf_mymin((ncx-1)*tf.ones_like(p[0,:]), tf.maximum(0.0, p[0,:]*ncx))
        p1 = tf_mymin((ncy-1)*tf.ones_like(p[1,:]), tf.maximum(0.0, p[1,:]*ncy))
        p2 = tf_mymin((ncz-1)*tf.ones_like(p[2,:]), tf.maximum(0.0, p[2,:]*ncz))

        # Initial cell index
        cell_idx = 6*(p0 + p1*ncx + p2*ncx*ncy)
        x = p[0,:]*ncx - p0
        y = p[1,:]*ncy - p1
        z = p[2,:]*ncz - p2
        
        # Find inner thetrahedron
        cell_idx = tf.where(tf.logical_and(tf.logical_and(tf.logical_and(x>y,x<=1-y),y<z),1-y>=z),
                            cell_idx+1,cell_idx)
        cell_idx = tf.where(tf.logical_and(tf.logical_and(tf.logical_and(x>=z,x<1-z),y>=z),y<1-z),
                            cell_idx+2,cell_idx)
        cell_idx = tf.where(tf.logical_and(tf.logical_and(tf.logical_and(x<=z,x>1-z),y<=z),y>1-z),
                            cell_idx+3,cell_idx)
        cell_idx = tf.where(tf.logical_and(tf.logical_and(tf.logical_and(x<y,x>=1-y),y>z),1-y<=z),
                            cell_idx+4,cell_idx)
        cell_idx = tf.where(tf.logical_and(tf.logical_and(tf.logical_and(x>=y,1-x<y),x>z),1-x<=z),
                            cell_idx+5,cell_idx)

        return cell_idx
    
#%%
if __name__ == '__main__':
    pass
    