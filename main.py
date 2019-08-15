# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:40:59 2018

@author: nsde
"""

if __name__ == "__main__":
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    from libcpab import Cpab

    T = Cpab([3, 3], backend='tensorflow', device='gpu', zero_boundary=True,
             volume_perservation=False, override=False)

    theta = T.sample_transformation()
    #theta = T.identity()

    grid = T.uniform_meshgrid([350, 350])

    grid_t = T.transform_grid(grid, theta)
    
    im = plt.imread('version1.4/data/cat.jpg')
    im = np.expand_dims(im, 0) / 255
    im = tf.cast(im, tf.float32)
    
    im_t = T.interpolate(im, grid_t, outsize=(350, 350))
    
    im = im.numpy()
    im_t = im_t.numpy()
    
    plt.figure()
    plt.imshow(im[0])
    plt.axis('off')    
    plt.tight_layout()
    plt.imsave('data/cat.jpg', im[0])
    
    plt.figure()
    plt.imshow(im_t[0])
    plt.axis('off')    
    plt.tight_layout()
    plt.imsave('data/deform_cat.jpg', im_t[0])

    T.visualize_vectorfield(theta)
    plt.axis([0,1,0,1])
    plt.tight_layout()
    plt.savefig('data/velocity_field.jpg')
    plt.show()
