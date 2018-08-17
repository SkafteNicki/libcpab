#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:46:23 2017

@author: nsde
"""

#%%
import tensorflow as tf

#%%
def tf_img_normalize(im):
    """ Normalize the image values to the 0..1 domain for each image in a batch"""
    with tf.name_scope('img_normalize'):
        im = im - tf.reduce_min(im, axis=[1,2,3], keepdims=True)
        im = im / tf.reduce_max(im, axis=[1,2,3], keepdims=True)
        return im

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
    """ Tensorflow implementation of np.repeat(x, n_repeats) """
    with tf.name_scope('repeat'):
        ones = tf.ones(shape=(n_repeats, ))
        rep = tf.transpose(tf.expand_dims(ones, 1), [1, 0])
        rep = tf.cast(rep, x.dtype)
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

#%%
def tf_shape_i(tensor, i):
    """ Extracts the i'th dimension of a tensor. If the dimension is unknown,
        the function will return a dynamic tensor else it will return a static
        number.
    """
    shape = tensor.get_shape().as_list()
    if shape[i]!=None:
        return shape[i]
    else:
        return tf.shape(tensor)[i]
    
    