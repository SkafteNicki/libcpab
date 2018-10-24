# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 08:09:00 2018

@author: nsde
"""
#%%
# This script is just intended for myself to do some debugging of the reposatory
# Please do not expect it to run 

#%%
import tensorflow as tf
from libcpab.tensorflow.cpab1d.transformer import tf_cpab_transformer_1D_cuda
from libcpab.tensorflow.cpab1d.transformer import tf_cpab_transformer_1D_pure
from libcpab.tensorflow import cpab
import numpy as np


T = cpab([10, ], zero_boundary=False, return_tf_tensors=True)

theta = 0.01*np.random.normal(size=(1, T.get_theta_dim()))
theta_tf = tf.cast(theta, tf.float32)
grid = T.uniform_meshgrid([518,])
res1 = tf_cpab_transformer_1D_cuda(grid, theta_tf)
res2 = tf_cpab_transformer_1D_pure(grid, theta_tf)
grad1 = tf.gradients(res1, theta_tf)
grad2 = tf.gradients(res2, theta_tf)


with tf.Session() as sess:
    with tf.device('/gpu:0'):
        result1 = sess.run(res1)
        result2 = sess.run(res2)
        gradient1 = sess.run(grad1)[0]
        gradient2 = sess.run(grad2)[0]
        print('Difference res:', np.linalg.norm(result1 - result2))
        print('Difference grad:', np.linalg.norm(gradient1 - gradient2))
        print(gradient1)
        print(gradient2)
        
T2 = cpab([10, ], zero_boundary=False, return_tf_tensors=False)
T2.visualize_vectorfield(theta[0])
