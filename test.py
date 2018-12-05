# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 08:09:00 2018

@author: nsde
"""
#%%
# This script is just intended for myself to do some debugging of the reposatory
# Please do not expect it to run 

#%%
import matplotlib.pyplot as plt
import torch
from libcpab.pytorch import cpab
T = cpab([3,3,3], zero_boundary=True, override=True)

img = plt.imread('data/cat.jpg') / 255
data = torch.zeros(1, 3, 100, 100, 100)
for i in range(100):
    data[:,:,i,:,:] = torch.tensor(img[125:225,125:225,:].transpose([2,0,1]))
    
theta = T.sample_transformation(1)
new_data = T.transform_data(data, theta, outsize=(100, 100, 100))

plt.figure()
plt.imshow(data[0,:,0,:,:].numpy().transpose([1,2,0]))
for i in range(20):
    plt.figure()
    plt.imshow(new_data[0,:,i,:,:].numpy().transpose([1,2,0]))

#%%
from libcpab.tensorflow import cpab
import matplotlib.pyplot as plt
import numpy as np
im = plt.imread('data/cat.jpg')
data = np.zeros((1, 100, 100, 100))
for i in range(100):
    data[0,i] = im[125:225, 125:225, 0]

T = cpab([3,3,3])
theta = 5*T.sample_transformation(1)
new_data = T.transform_data(data, theta)
