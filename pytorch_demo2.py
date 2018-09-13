#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 08:20:29 2018

@author: nsde
"""

#%%
from libcpab.pytorch import cpab
import matplotlib.pyplot as plt
import numpy as np
import torch

#%%
if __name__ == '__main__':
    # Load some data
    data = plt.imread('data/cat.jpg') / 255
    data = np.expand_dims(data, 0) # create batch effect
    data = torch.Tensor(data).permute(0,3,1,2)
    
    # Create transformer class
    T1 = cpab(tess_size=[3,3])
    
    # Sample random transformation
    theta_true = 0.5*T1.sample_transformation(1)
    
    # Transform the images
    transformed_data = T1.transform_data(data, theta_true, outsize=(350, 350))

    # Now, create pytorch procedure that enables us to estimate the transformation
    # we have just used for transforming the data
    T2 = cpab(tess_size=[3,3], device='cpu')
    theta_est = T2.identity(1, epsilon=1e-4)
    theta_est.requires_grad = True
    
    # Pytorch optimizer
    optimizer = torch.optim.Adam([theta_est], lr=0.1)
    
    # Optimization loop
    maxiter = 100
    for i in range(maxiter):
        trans_est = T2.transform_data(data, theta_est, outsize=(350, 350))
        loss = (transformed_data.to(trans_est.device) - trans_est).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Iter', i, ', Loss', np.round(loss.item(), 4), ', ||theta_true - theta_est||: ',
              np.linalg.norm((theta_true-theta_est.cpu().detach()).numpy().round(4)))
    
    # Show the results
    plt.subplots(1,3, figsize=(10, 15))
    plt.subplot(1,3,1)
    plt.imshow(data.permute(0,2,3,1).numpy()[0])
    plt.axis('off')
    plt.title('Source')
    plt.subplot(1,3,2)
    plt.imshow(transformed_data.permute(0,2,3,1).cpu().numpy()[0])
    plt.axis('off')
    plt.title('Target')
    plt.subplot(1,3,3)
    plt.imshow(trans_est.permute(0,2,3,1).cpu().detach().numpy()[0])
    plt.axis('off')
    plt.title('Estimate')