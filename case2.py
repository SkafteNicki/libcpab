# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 10:45:45 2018

@author: nsde
"""

#%%
from libcpab import cpab
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#%%
if __name__ == '__main__':
    # Load some data
    data = plt.imread('data/cat.jpg') / 255
    data = np.expand_dims(data, 0) # create batch effect
    
    # Create transformer class
    T = cpab(tess_size=[3,3])
    
    # Sample 9 random transformation
    theta_true = T.sample_transformation(1)
    
    # Transform the images
    transformed_data = T.transform_data(data, theta_true)

    # Now try to estimate the transformation
    T = cpab(tess_size=[3,3], return_tf_tensors=True)
    theta_est = tf.Variable(initial_value=T.identity(1)+1e-4)
    trans_est = T.transform_data(data, theta_est)
    loss = tf.reduce_mean(tf.pow(transformed_data - trans_est, 2))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    trainer = optimizer.minimize(loss)
    
    maxiter = 100
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(maxiter):
        _, l, t_est = sess.run([trainer, loss, theta_est])
        print(i, l)
    