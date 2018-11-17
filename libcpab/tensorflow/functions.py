# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 16:01:52 2018

@author: nsde
"""

#%%
import tensorflow as tf

#%%
def atype():
    return [tf.python.ops.variables.RefVariable,
            tf.python.framework.ops.Tensor]
    
#%%
def sample_transformation(n_sample=1, mean=None, cov=None):
    pass

#%%
def identity(d, n_sample=1, epsilon=0):
    return tf.zeros((d, n_sample), dtype=tf.float32)