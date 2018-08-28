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
    import scipy.io
    import numpy as np
    import torch
    
    data = scipy.io.loadmat('data/mri_scan.mat')['D']
    data = torch.Tensor(data)[None,None,:,:,:].permute(0,1,4,3,2)
    T = cpab([2,2,2])
    g = T.uniform_meshgrid([128, 128, 27])
    theta = T.identity(1)
    gt = T.transform_grid(g, theta)
    data_new = T.interpolate(data, gt, outsize=(128, 128, 27))
#    im_new = T.transform_data(im2, theta, outsize=(350, 350))
#    image = im_new.permute(0,2,3,1)
#    image = image - image.min()
#    image = image / image.max()
#    
#    plt.imshow(image[0].numpy())
    
    
    
    
    
    