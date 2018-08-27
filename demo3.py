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
    y1 = x**2
    y2 = (x+0.1)**2
    
    # Create transformer
    T = cpab(tess_size=[5,])
    T.fix_data_size([50,]) # fix the data size for speed
    
    # Lets do some sampling
    maxiter = 200
    current_samples = T.identity(1)
    current_error = np.linalg.norm(y1 - y2)
    all_samples = [ ]
    for i in range(maxiter):
        # Sample random transformations
        theta = T.sample_transformation(1, mean=np.squeeze(current_samples))
        y1_trans = T.transform_data(np.expand_dims(y1,0), theta)

        # Compare to current 
        new_error = np.linalg.norm(y1_trans - y2)
        ratio = np.exp(-new_error)/np.exp(-current_error)
        if ratio > 1:
            current_sample = theta
            all_samples.append(theta)
            print(i, 'accept', ratio, new_error, current_error)
        else:
            print(i, 'reject', ratio)
    samples = np.array(all_samples)
#    mean = np.mean(samples, axis=0)
#    print('theta estimate', mean)
#    
    y1_mean = T.transform_data(np.expand_dims(y1, 0), samples[-1])
    plt.plot(y1, '-r')
    plt.plot(y1_mean[0], '-b')
    plt.plot(y2, 'g-')
