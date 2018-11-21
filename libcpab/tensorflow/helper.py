# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 09:35:18 2018

@author: nsde
"""

#%%
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
