# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 16:01:52 2018

@author: nsde
"""

#%%
import torch
from .interpolation import interpolate
from .transformer import CPAB_transformer as transformer
from .findcellidx import findcellidx
from ..core.utility import load_basis_as_struct

#%%
def to(x):
    return torch.Tensor(x)

#%%
def tonumpy(x):
    return x.cpu().numpy()

#%%
def type():
    return [torch.Tensor]

#%%
def pdist(mat):
    norm = torch.sum(mat * mat, 1)
    norm = torch.reshape(norm, (-1, 1))
    D = norm - 2*mat.mm(mat.t()) + norm.t()
    return D

#%%
def norm(x):
    return torch.norm(x)

#%%
def sample_transformation(d, n_sample=1, mean=None, cov=None, device='cpu'):
    device = torch.device('cpu') if device=='cpu' else torch.device('cuda')
    mean = torch.zeros(d, dtype=torch.float32, device=device) if mean is None else mean
    cov = torch.eye(d, dtype=torch.float32, device=device) if cov is None else cov
    distribution = torch.distributions.MultivariateNormal(mean, cov)
    return distribution.sample((n_sample,))

#%%
def identity(d, n_sample=1, epsilon=0, device='cpu'):
    assert epsilon>=0, "epsilon need to be larger than 0"
    device = torch.device('cpu') if device=='cpu' else torch.device('cuda')
    return torch.zeros(n_sample, d, dtype=torch.float32, device=device) + epsilon

#%%
def uniform_meshgrid(ndim, domain_min, domain_max, n_points, device='cpu'):
    device = torch.device('cpu') if device=='cpu' else torch.device('cuda')
    lin = [torch.linspace(domain_min[i], domain_max[i], n_points[i], 
                          device=device) for i in range(ndim)]
    mesh = torch.meshgrid(lin)
    grid = torch.cat([g.reshape(1,-1) for g in mesh], dim=0)
    return grid

#%%
def calc_vectorfield(grid, theta):
    # Load parameters
    params = load_basis_as_struct()
    
    # Calculate velocity fields
    Avees = torch.matmul(params.basis, theta)
    As = torch.reshape(Avees, (params.nC, *params.Ashape))
    
    # Find cell index
    idx = findcellidx(params.ndim, grid, params.nc)
    
    # Do indexing
    Aidx = As[idx]
    
    # Convert to homogeneous coordinates
    grid = torch.cat((grid, torch.ones(1, grid.shape[1])), dim=0)
    grid = grid[None].permute(2,1,0)
    
    # Do matrix multiplication
    v = torch.matmul(Aidx, grid)
    return v[:,:,0].t()