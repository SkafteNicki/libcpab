# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 10:37:45 2018

@author: nsde
"""

#%%
import torch
from .torch_funcs import torch_repeat_matrix, torch_expm

#%%
class CPAB_transformer(torch.nn.Module):
    def __init__(self, params, findcellidx_func, device):
        super(CPAB_transformer, self).__init__()
        self.params = params
        self.findcellidx = findcellidx_func
        self.device = device
        
    def __call__(self, points, theta):
        # Problem sizes
        n_points = points.shape[1]
        n_theta = theta.shape[0]
        
        # Transform points
        ones = torch.ones(n_theta, 1, n_points).to(self.device)
        newpoints = torch_repeat_matrix(points, n_theta) # [n_theta, ndim, n_points]
        newpoints = torch.cat([newpoints, ones], dim=1)
        newpoints = newpoints.permute(0, 2, 1).reshape(-1, self.params.ndim+1)
        newpoints = newpoints[:,:,None] # [n_theta * ndim, ndim+1, 1]
     
        # Get velocity fields
        B = torch.Tensor(self.params.basis).to(self.device)
        Avees = torch.matmul(B, theta.t())
        As = Avees.t().reshape(n_theta*self.params.nC, *self.params.Ashape)
        zero_row = torch.zeros(n_theta*self.params.nC, 1, self.params.ndim+1).to(self.device)
        AsSquare = torch.cat([As, zero_row], dim=1)
        
        # Take matrix exponential
        dT = 1.0 / self.params.nstepsolver
        Trels = torch_expm(dT*AsSquare)

        # Batch index to add to correct for the batch effect
        batch_idx = self.params.nC * (torch.ones(n_points, n_theta, dtype=torch.int64) * 
                                      torch.arange(n_theta)).t().flatten()
        batch_idx = batch_idx.to(self.device)
        
        # Intergrate velocity field
        for i in range(self.params.nstepsolver):
            # Find cell index and correct for batching effect
            idx = self.findcellidx(newpoints, *self.params.nc) + batch_idx
            
            # Gather the correct transformations
            Tidx = Trels[idx]
            
            # Transform points
            newpoints = torch.matmul(Tidx, newpoints)
        
        # Reshape to the right format
        newpoints = newpoints.squeeze()[:,:self.params.ndim].t()
        newpoints = newpoints.reshape(self.params.ndim, n_theta, n_points).permute(1,0,2)
        return newpoints
    