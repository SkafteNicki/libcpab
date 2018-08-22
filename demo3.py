# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 09:57:26 2018

@author: nsde
"""
#%%
from libcpab import cpab
import numpy as np
import matplotlib.pyplot as plt

#%%
if __name__ == '__main__':
    # Lets create some 1D data
    x = np.linspace(0, 1, 50)    
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # Create transformer
    T = cpab(tess_size=[5,])
    T.fix_data_size([50,]) # fix the data size for speed
    
    # Lets do some sampling
    maxiter = 100
    current_samples = T.identity(1)
    current_error = np.linalg.norm(y1 - y2)
    all_samples = [ ]
    for i in range(maxiter):
        # Sample random transformations
        theta = T.sample_transformation(1, mean=current_samples)
        y1_trans = T.transform_data(np.expand_dims(y1,0), theta)
        
        # Compare to current 
        new_error = np.linalg.norm(y1_trans - y2)
        
        if np.log(np.random.uniform()) < (new_error - current_error):
            current_sample = theta
            all_samples.append(theta)
    
    samples = np.array(all_samples)
    mean = np.mean(samples, axis=0)
    print('theta estimate', mean)
    
    y1_mean = T.transform_data(np.expand_dims(y1, 0), np.expand_dims(mean, 0))
    plt.plot(y1_mean, '-b')
    plt.plot(y2, 'g-')