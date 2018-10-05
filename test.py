# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 08:09:00 2018

@author: nsde
"""
#%%
# This script is just intended for myself to do some debugging of the reposatory
# Please do not expect it to run 

#%%
from libcpab.pytorch import cpab
import torch
import numpy as np
import csv, sys

#%%
def LoadBrain(filename): 
    with open(filename, newline='') as f:
        data=[]
        for row in csv.reader(f,quoting=csv.QUOTE_NONNUMERIC,delimiter=','):
            data.append(row)
        f.close
    data=data[0]

    dim=[0,0,0]
    for i in range(0,3):
        dim[i]=int(data[i])

    data=np.reshape(data[3:],dim)
    return data

#%%
def Normalize(data):
    data = data-data.mean()
    data = data / data.std()
    return data

#%%
def SliceBrain(data,slice_size):
    import numpy as np
    
    dim_brain = np.array(np.shape(data))
    dim_slice = np.array(np.shape(slice_size)[0])
        
    if len(dim_brain) != dim_slice:
        print('Data and slize_size must have the same dimensions')
        return
    if (dim_brain%2).sum() != 0:
        print('Brain dimensions must be even')
        return
    
    if (np.array(slice_size)%2).sum() != 0:
        print('slice_size dimensions must be even')
    
    if len(dim_brain) == 2:
        mid = dim_brain/2
        upper = [0,0]
        lower = [0,0]
        for i in range(2):
            upper[i]=int(mid[i]+slice_size[i]/2)
            lower[i]=int(mid[i]-slice_size[i]/2)
        
        return data[lower[0]:upper[0],lower[1]:upper[1]]
    
    
    if len(dim_brain) == 3:
        mid = dim_brain/2
        upper = [0,0,0]
        lower = [0,0,0]
        for i in range(3):
            upper[i]=int(mid[i]+slice_size[i]/2)
            lower[i]=int(mid[i]-slice_size[i]/2)
        
        return data[lower[0]:upper[0],lower[1]:upper[1],lower[2]:upper[2]]

#%%
print('Number of arguments:', len(sys.argv), 'arguments.')
print( 'Argument List:', str(sys.argv))

s1=int(sys.argv[1])
s2=int(sys.argv[2])
s3=int(sys.argv[3])
maxiter= int(sys.argv[4])

print('slice1=', str(s1),'slice2=',str(s2), 'slice3=',str(s3), 'maxiter:',str(maxiter))

#%%
if __name__ == '__main__':
    
    slice_size = [s1,s2,s3]
    print('\n--- Loading data ---\n')
    # Load some data
    data1 = LoadBrain('data/IBSR_01_ana_strip.csv')
    print(data1.shape)    
    # Extracting a (s1,s2,s3) patch of the data
    data1 = SliceBrain(data1,slice_size)
    data1 = Normalize(data1)
    data1 = np.expand_dims(data1, 0) #Create color channel effect
    data1 = np.expand_dims(data1, 0) # create batch effect
    data1 = torch.Tensor(data1).cuda()
    print(data1.shape)    
    # Loading another img.
    data2 = LoadBrain('data/IBSR_02_ana_strip.csv')
    print(data2.shape)
    # Extracting a (s1,s2,s3) patch of the data
    data2 = SliceBrain(data2,slice_size)
    data2 = Normalize(data2)
    data2 = np.expand_dims(data2, 0) #Create color channel effect
    data2 = np.expand_dims(data2, 0) # create batch effect
    data2 = torch.Tensor(data2).cuda()
    print(data1.shape)
    print('--- Setting up tessalation and optimization parameters ---\n')
    # Now, create pytorch procedure that enables us to estimate the transformation
    # we have just used for transforming the data
    T2 = cpab(tess_size=[3,3,3], device='gpu')
    theta_est = T2.identity(1, epsilon=1e-4)
    theta_est.requires_grad = True
    
    # Pytorch optimizer
    optimizer = torch.optim.Adam([theta_est], lr=0.1)
    
    # Optimization loop
    losses = []
    for i in range(maxiter):
        trans_est = T2.transform_data(data1, theta_est, outsize=slice_size)
        print(trans_est.shape)
        loss = (data2 - trans_est).abs().pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Iter', i, ', Loss', np.round(loss.item(), 4))
        losses.append(loss.item())
        del loss
        
    print('\n --- Training is done ---\n')
