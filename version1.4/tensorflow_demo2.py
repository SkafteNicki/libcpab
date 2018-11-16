# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 10:45:45 2018

@author: nsde
"""

#%%
from libcpab.tensorflow import cpab
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#%%
if __name__ == '__main__':
    # Load some data
    data = plt.imread('data/cat.jpg') / 255
    data = np.expand_dims(data, 0) # create batch effect
    
    # Create transformer class
    T1 = cpab(tess_size=[3,3])
    
    # Sample random transformation
    theta_true = 0.5*T1.sample_transformation(1)
    
    # Transform the images
    transformed_data = T1.transform_data(data, theta_true)

    # Now, create tensorflow graph that enables us to estimate the transformation
    # we have just used for transforming the data
    T2 = cpab(tess_size=[3,3], return_tf_tensors=True)
    theta_est = tf.Variable(initial_value=T1.identity(1))
    trans_est = T2.transform_data(data, theta_est)
    loss = tf.reduce_mean(tf.pow(transformed_data - trans_est, 2))
    
    # Tensorflow optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    trainer = optimizer.minimize(loss)
    
    # Optimization loop
    maxiter = 100
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(maxiter):
        _, l, t_est = sess.run([trainer, loss, theta_est])
        print('Iter: ', i, ', Loss: ', l.round(4), ', ||theta_true - theta_est||: ',
              np.linalg.norm(theta_true - t_est).round(4))
    est_trans = T1.transform_data(data, t_est)
    
    # Show the results
    plt.subplots(1,3, figsize=(10, 15))
    plt.subplot(1,3,1)
    plt.imshow(data[0])
    plt.axis('off')
    plt.title('Source')
    plt.subplot(1,3,2)
    plt.imshow(transformed_data[0])
    plt.axis('off')
    plt.title('Target')
    plt.subplot(1,3,3)
    plt.imshow(est_trans[0])
    plt.axis('off')
    plt.title('Estimate')
    
    