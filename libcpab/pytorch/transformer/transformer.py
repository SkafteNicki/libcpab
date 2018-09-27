#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 11:46:55 2018

@author: nsde
"""

#%%
import torch
from torch.utils.cpp_extension import load
import warnings

#from ..torch_funcs import torch_repeat_matrix, torch_expm
from libcpab.pytorch.torch_funcs import torch_repeat_matrix, torch_expm

#%%
class _notcompiled:
    # Small class, with structure similear to the compiled modules we can default
    # to. The class will never be called but the program can compile
    def __init__(self):
        def f(*args):
            return None
        self.forward = f
        self.backward = f
        
#%%
# Compile cpu source
#try:
#    with warnings.catch_warnings(record=True) as w:
cpab_cpu = load(name = 'cpab_cpu',
                sources = ['CPAB_ops.cpp'],
                verbose=True)

cpab_gpu = load(name = 'cpab_gpu',
                sources = ['CPAB_ops_cuda.cpp', 'CPAB_ops_cuda_kernel.cu'],
                verbose=True)
#    cpu_succes = True
#except:
#    square_cpu = _notcompiled()
#    cpu_succes = False

#%%
class _IntegrationFunction(torch.autograd.Function):
    # Function that connects the forward pass to the backward pass
    @staticmethod
    def forward(ctx, points, theta):
        assert points.device == theta.device, ''' points and theta is not placed
            on the same device'''
        device = points.device
        # Problem size
        n_points = points.shape[1]
        n_theta = theta.shape[0]
        
        # Transform points
        
#        if points.is_cuda and theta.is_cuda:
#            output = cpab_gpu.forward(input)
#        else:
#            output = cpab_cpu.forward(input)
#        ctx.save_for_backward(input)
#        return output
        output = cpab_cpu.forward()
#        
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_points, grad_theta):
        return grad_points, grad_theta

def _fast_transformer(points, theta):
    return _IntegrationFunction.apply(points, theta)

#%%
def _slow_transformer(points, theta):
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
        del idx, Tidx
    
    # Reshape to the right format
    newpoints = newpoints.squeeze()[:,:self.params.ndim].t()
    newpoints = newpoints.reshape(self.params.ndim, n_theta, n_points).permute(1,0,2)
    return newpoints
    
#%%
def transformer(points, theta):
    ''' This wrapper function, combines the fast and slow transformer into one
        function. If the fast versions have been successfully compiled, then these
        will be used, else default to the slow versions '''
    if points.is_cuda and theta.is_cuda:
        if gpu_succes:
            output = _fast_transformer(input)
        else:
            output = _slow_transformer(input)
    else:
        if cpu_succes:
            output = _fast_transformer(input)
        else:
            output = _slow_transformer(input)
    return output        
    
