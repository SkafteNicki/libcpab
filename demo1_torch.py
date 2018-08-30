#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 07:54:48 2018

@author: nsde
"""

#%%
from libcpab.develop.cpab import cpab
from libcpab.helper.utility import show_images
import matplotlib.pyplot as plt
import numpy as np
import torch

#%%
if __name__ == '__main__':
    # Number of transformed samples 
    N = 9
    
    # Load some data
    data = plt.imread('data/cat.jpg') / 255
    data = np.tile(np.expand_dims(data, 0), [N,1,1,1]) # create batch of data
    
    # Convert to torch tensor and torch format [n_batch, n_channels, width, height]
    data = torch.Tensor(data).permute(0,3,1,2)
    
    # Define transformer class
    T = cpab(tess_size=[3,3], device='cpu')
    
    # Sample random transformation
    theta = 0.5*T.sample_transformation(N)
    
    # Transform the images
    transformed_data = T.transform_data(data, theta, outsize=(350, 350))

    # Get the corresponding numpy arrays in correct format
    transformed_data = transformed_data.permute(0, 2, 3, 1).numpy()

    # Show transformed samples
    show_images(transformed_data)