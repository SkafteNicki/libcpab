# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 10:26:30 2018

@author: nsde
"""

#%%
import torch

#%%
def interpolate(ndim, data, grid, outsize):
    if ndim==1: return interpolate1D(data, grid, outsize)
    elif ndim==2: return interpolate2D(data, grid, outsize)
    elif ndim==3: return interpolate3D(data, grid, outsize)

#%%    
def interpolate1D(data, grid, outsize):
    n_batch = data.shape[0]
    in_size = data.shape[1]
    out_size = grid.shape[2]
    max_x = in_size - 1
    
    # Extract points
    x = grid[:,0].flatten()
    
    # Scale to domain
    x = x*(in_size-1)
    
    # Do sampling
    x0 = torch.floor(x).type(torch.int64)
    x1 = x0 + 1
    
    # Clip values
    x0 = torch.clamp(x0, 0, max_x)
    x1 = torch.clamp(x1, 0, max_x)
    
    # Take care of batch effect
    base = (torch.arange(n_batch)*in_size).repeat(out_size,1).t().flatten()
    base = base.to(grid.device)
    idx1 = base + x0
    idx2 = base + x1
    
    # Lookup values
    data_flat = data.flatten()
    i1 = torch.gather(data_flat, 0, idx1).type(torch.float32)
    i2 = torch.gather(data_flat, 0, idx2).type(torch.float32)
        
    # Convert to floats
    x0 = x0.type(torch.float32)
    x1 = x1.type(torch.float32)
        
    # Interpolation weights
    w1 = (x - x0)
    w2 = (x1 - x)
        
    # Do interpolation
    new_data = w1*i1 + w2*i2
        
    # Reshape and return
    new_data = torch.reshape(new_data, (n_batch, out_size))
    return new_data

#%%    
def interpolate2D(data, grid, outsize):
    grid = grid.permute(0,2,1).reshape(data.shape[0], *outsize, 2).permute(0,2,1,3)
    interpolated = torch.nn.functional.grid_sample(data, grid)
    return interpolated

#%%    
def interpolate3D(data, grid, outsize):
    grid = grid.permute(0,2,1).reshape(data.shape[0], *outsize, 3).permute(0,3,2,1,4)
    interpolated = torch.nn.functional.grid_sample(data, grid)
    return interpolated
