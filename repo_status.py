# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:50:53 2018

@author: nsde
"""

# TODO: test interpolation for 3D

#%%
if __name__ == '__main__':
    from libcpab.develop.cpab import cpab
    import matplotlib.pyplot as plt
    import torch
    
    im1 = plt.imread('data/cat.jpg')
    im2 = torch.Tensor(im1)[None,:,:,:].permute(0, 3, 1, 2)
    
    T = cpab([2,2])
    g = T.uniform_meshgrid([350, 350])
    theta = T.identity(1)
    theta = T.sample_transformation(1)
    gt = T.transform_grid(g, theta)
    
    #gt = gt.permute(0,2,1).reshape(1, 350, 350, 2).permute(0,2,1,3)
    #grid.permute(0,)
    #im_new = T.interpolate(im2, gt, outsize=(350, 350))
    im_new = T.transform_data(im2, theta, outsize=(350, 350))
    image = im_new.permute(0,2,3,1)
    image = image - image.min()
    image = image / image.max()
    
    plt.imshow(image[0].numpy())
    
    
    
    
    
    