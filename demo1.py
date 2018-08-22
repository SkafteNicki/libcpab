# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 10:45:21 2018

@author: nsde
"""

#%%
from libcpab import cpab
from libcpab.helper.utility import show_images
import matplotlib.pyplot as plt
import numpy as np

#%%
if __name__ == '__main__':
    # Number of transformed samples 
    N = 9
    
    # Load some data
    data = plt.imread('data/cat.jpg') / 255
    data = np.tile(np.expand_dims(data, 0), [N,1,1,1]) # create batch of data
    
    # Define transformer class
    T = cpab(tess_size=[3,3])
    
    # Sample random transformation
    theta = 0.5*T.sample_transformation(N)
    
    # Transform the images
    transformed_data = T.transform_data(data, theta)

    # Show transformed samples
    show_images(transformed_data)