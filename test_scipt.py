# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:50:53 2018

@author: nsde
"""
# TODO: fix problem with gradient of pure tf version (probably all dim)

if __name__ == '__main__':
    from libcpab import cpab
    from libcpab.helper.utility import show_images, load_basis
    from libcpab.helper.tf_funcs import tf_shape_i, tf_repeat_matrix
    from libcpab.helper.tf_expm import tf_expm3x3
    from libcpab.helper.tf_findcellidx import tf_findcellidx_2D
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    
    # Load some data
    N = 1
    data = plt.imread('data/cat.jpg') / 255
    data = np.tile(np.expand_dims(data, 0), [N,1,1,1]) # create batch of data
        
    
    T = cpab(tess_size=[3,3], return_tf_tensors=True)
    points = T.uniform_meshgrid([350, 350])
    #theta = T.sample_transformation(1)
    theta = tf.Variable(initial_value=T.identity(1))
    
    dT = 1.0 / tf.cast(50, tf.float32)
    tess = load_basis()
    B = tf.cast(tess['basis'], tf.float32)
    B = tf_repeat_matrix(B, 1)
    Avees = tf.matmul(B, tf.expand_dims(theta, 2))
    As = tf.reshape(Avees, shape = (36, *tess['Ashape'])) # format [n_theta * nC, 2, 3]
    Trels = tf_expm3x3(dT*As)

    
    points_new = T.transform_grid(points, theta)
    new_data = T.interpolate(data, points_new)
    
    loss = tf.reduce_mean(tf.pow(data - new_data, 2.0))
    
    g0 = tf.gradients([points_new], theta)[0]
    g1 = tf.gradients([new_data], theta)[0]
    g2 = tf.gradients([loss], theta)[0]
    g3 = tf.gradients([Trels], theta)[0]

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    
    
    
    
    
    
    