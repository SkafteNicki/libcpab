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

    T = Cpab([3, 3], backend='numpy', device='cpu', zero_boundary=True,
             volume_perservation=False, override=False)

    theta = 0.5*T.sample_transformation()
    #theta = T.identity()

    grid = T.uniform_meshgrid([350, 350])

    grid_t = T.transform_grid(grid, theta)
    
    im = plt.imread('version1.4/data/cat.jpg')
    im = np.expand_dims(im, 0) / 255
    #im = tf.cast(im, tf.float32)
    
    im_t = T.interpolate(im, grid_t, outsize=(350, 350))
    
    plt.figure()
    plt.imshow(im[0])
    plt.axis('off')    
    plt.tight_layout()
    #plt.imsave('cat.jpg', im[0].numpy())
    
    plt.figure()
    plt.imshow(im_t[0])
    plt.axis('off')    
    plt.tight_layout()
    #plt.imsave('deform_cat.jpg', im_t[0].numpy())

    T.visualize_vectorfield(theta)
    plt.axis([0,1,0,1])
    plt.tight_layout()
    #plt.savefig('velocity_field.jpg')
    plt.show()

    # %% numpy test)
#    im = np.tile(plt.imread('cat.jpg')[None] / 255, (2, 1, 1, 1))
#
#    T = cpab([2,], backend='numpy', device='cpu', zero_boundary=True,
#             volume_perservation=False, override=False)
#
#    theta = T.sample_transformation()
#    theta = np.tile(theta, (2, 1))
#
#    grid = T.uniform_meshgrid([350, 350])
#    grid = np.tile(grid[None], (2, 1, 1))
#    grid[0] += 0.1
#
#    grid_t = T.transform_grid(grid, theta)
#
#    im_t = T.interpolate(im, grid_t, outsize=(350, 350))
#
#    plt.figure();plt.imshow(im_t[0])
#    plt.figure();plt.imshow(im_t[1])
##
# %% pytorch test
#    im = torch.tensor(plt.imread('cat.jpg')[None] / 255).repeat(2, 1, 1, 1)
#    im = im.to(torch.float32).permute(0, 3, 1, 2)
#
#    T = cpab([3,3], backend='pytorch', device='gpu', zero_boundary=True,
#             volume_perservation=False, override=False)
#
#    theta = T.sample_transformation()
#    theta = theta.repeat(2,1)
#
#    grid = T.uniform_meshgrid([350, 350])
#    grid = grid[None].repeat(2, 1, 1)
#    grid[0] += 0.1
#
#    grid_t = T.transform_grid(grid, theta)
#
#    im_t = T.interpolate(im.cuda(), grid_t, outsize=(350, 350))
#
#    plt.figure();plt.imshow(im_t[0].permute(1, 2, 0).cpu().numpy())
#    plt.figure();plt.imshow(im_t[1].permute(1, 2, 0).cpu().numpy())
#
#    #%% tensorflow test
#    im = tf.cast(plt.imread('cat.jpg')[None] / 255, tf.float32)
#

#
##    theta = T.sample_transformation()
##    theta = theta.repeat(2,1)
##
#    grid = T.uniform_meshgrid([350, 350])
#    grid = grid[None].repeat(2, 1, 1)
#    grid[0] += 0.1

#    grid_t = T.transform_grid(grid, theta)

#    im_t = T.interpolate(im.cuda(), grid_t, outsize=(350, 350))
#
#    plt.figure();plt.imshow(im_t[0].permute(1, 2, 0).cpu().numpy())
#    plt.figure();plt.imshow(im_t[1].permute(1, 2, 0).cpu().numpy())
