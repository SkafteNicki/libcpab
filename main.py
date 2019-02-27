# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:40:59 2018

@author: nsde
"""

if __name__ == "__main__":
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from libcpab import cpab
    
#    #%% numpy test
#    im = np.tile(plt.imread('cat.jpg')[None] / 255, (2, 1, 1, 1))
#    
#    T = cpab([3,3], backend='numpy', device='cpu', zero_boundary=True, 
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
    
    #%% pytorch test
    im = torch.tensor(plt.imread('cat.jpg')[None] / 255).repeat(2, 1, 1, 1)
    im = im.to(torch.float32).permute(0, 3, 1, 2)
    
    T = cpab([3,3], backend='pytorch', device='gpu', zero_boundary=True,
             volume_perservation=False, override=False)
    
    theta = T.sample_transformation()
    theta = theta.repeat(2,1)
    
    grid = T.uniform_meshgrid([350, 350])
    grid = grid[None].repeat(2, 1, 1)
    grid[0] += 0.1
    
    grid_t = T.transform_grid(grid, theta)
    
    im_t = T.interpolate(im.cuda(), grid_t, outsize=(350, 350))
    
    plt.figure();plt.imshow(im_t[0].permute(1, 2, 0).cpu().numpy())
    plt.figure();plt.imshow(im_t[1].permute(1, 2, 0).cpu().numpy())
    
