# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:28:55 2018

@author: nsde
"""

#%%
import tensorflow as tf
from tensorflow.python.framework import function

from ..helper.utility import get_dir
from ..helper.utility import gpu_support as _gpu_support
from sys import platform as _platform

#%% Load dynamic module
def load_dynamic_modules():
    dir_path = get_dir(__file__)
    transformer_module = tf.load_op_library(dir_path + '/./CPAB_ops.so')
    transformer_op = transformer_module.calc_trans
    grad_op = transformer_module.calc_grad
    
    return transformer_op, grad_op

if _platform == "linux" or _platform == "linux2" or _platform == "darwin":    
    transformer_op, grad_op = load_dynamic_modules()

#%%
def _calc_trans(points, theta, tess):
    """ """
    with tf.name_scope('calc_trans'):
        pass

#%%        
def _calc_grad(op, grad):
    """ """
    with tf.name_scope('calc_grad'):
        pass

#%%
@function.Defun(tf.float32, tf.float32, tf.variant, func_name='tf_CPAB_transformer_3D', python_grad_func=_calc_grad)
def tf_cpab_transformer_3D_cuda(points, theta, tess):
    """ """
    return _calc_trans(points, theta, tess)

#%%
def tf_cpab_transformer_3D_pure(points, theta, tess):
    """ """
    with tf.name_scope('CPAB_transformer'):
        pass

#%%
_gpu = _gpu_support()
if _gpu:
    tf_cpab_transformer_3D = tf_cpab_transformer_3D_cuda
else:
    tf_cpab_transformer_3D = tf_cpab_transformer_3D_pure