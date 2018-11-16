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
from ..torch_funcs import torch_repeat_matrix, torch_expm, torch_findcellidx
from ...helper.utility import load_basis_as_struct, get_dir

#%%
class _notcompiled:
    # Small class, with structure similear to the compiled modules we can default
    # to. The class will never be called but the program can compile at run time
    def __init__(self):
        def f(*args):
            return None
        self.forward = f
        self.backward = f
        
#%%
# Try to compile the cpu and gpu version. The warning statement will fix 
# ABI-incompatible warning I am getting. Is not nessesary with a newer version
# of gcc compiler. The try-statement makes sure that we default to a slower
# version if we fail to compile one of the versions     
_verbose = False
_use_slow = False
_use_numeric = False
_dir = get_dir(__file__)        

# Jit compile cpu source
try:
    with warnings.catch_warnings(record=True):        
        cpab_cpu = load(name = 'cpab_cpu',
                        sources = [_dir + '/CPAB_ops.cpp'],
                        verbose=_verbose)
    _cpu_succes = True
    if _verbose:
        print(70*'=')
        print('succesfully compiled cpu source')
        print(70*'=')
except Exception as e:
    cpab_cpu = _notcompiled()
    _cpu_succes = False
    if _verbose:
        print(70*'=')
        print('Unsuccesfully compiled cpu source')
        print('Error was: ')
        print(e)
        print(70*'=')

# Jit compile gpu source
try:
    with warnings.catch_warnings(record=True):
        cpab_gpu = load(name = 'cpab_gpu',
                        sources = [_dir + '/CPAB_ops_cuda.cpp', 
                                   _dir + '/CPAB_ops_cuda_kernel.cu'],
                        verbose=_verbose)
    _gpu_succes = True
    if _verbose:
        print(70*'=')
        print('succesfully compiled gpu source')
        print(70*'=')
except Exception as e:
    cpab_gpu = _notcompiled()
    _gpu_succes = False
    if _verbose:
        print(70*'=')
        print('Unsuccesfully compiled gpu source')
        print('Error was: ')
        print(e)
        print(70*'=')

#%%
class _CPABFunction(torch.autograd.Function):
    # Function that connects the forward pass to the backward pass
    @staticmethod
    def forward(ctx, points, theta):
        assert points.device == theta.device, '''points are on device %s and 
            theta are on device %s. Please make sure that they are on the 
            same device''' % (str(points.device), str(theta.device))
        device = points.device
        params = load_basis_as_struct()
        
        # Problem size
        n_theta = theta.shape[0]
        
        # Get velocity fields
        B = torch.Tensor(params.basis).to(device)
        Avees = torch.matmul(B, theta.t())
        As = Avees.t().reshape(n_theta*params.nC, *params.Ashape)
        zero_row = torch.zeros(n_theta*params.nC, 1, params.ndim+1).to(device)
        AsSquare = torch.cat([As, zero_row], dim=1)
        
        # Take matrix exponential
        dT = 1.0 / params.nstepsolver
        Trels = torch_expm(dT*AsSquare)
        Trels = Trels[:,:params.ndim,:].view(n_theta, params.nC, *params.Ashape)
        
        # Convert to tensor
        nstepsolver = torch.tensor(params.nstepsolver, dtype=torch.int32, 
                                   device=device).to(device)
        nc = torch.tensor(params.nc, dtype=torch.int32, device=device)
        
        # Call integrator
        if points.is_cuda:
            newpoints = cpab_gpu.forward(points.contiguous(), 
												Trels.contiguous(), 
												nstepsolver.contiguous(), 
												nc.contiguous())
        else:            
            newpoints = cpab_cpu.forward(points.contiguous(), 
												Trels.contiguous(), 
												nstepsolver.contiguous(), 
												nc.contiguous())
            
        # Save of backward
        Bs = B.t().view(-1, params.nC, *params.Ashape)
        As = As.view(n_theta, params.nC, *params.Ashape)
        ctx.save_for_backward(points, theta, As, Bs, nstepsolver, nc)
        # Output result
        return newpoints

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad): # grad [n_theta, ndim, n]
        # Grap input
        points, theta, As, Bs, nstepsolver, nc = ctx.saved_tensors
        
        # Call integrator, gradient: [d, n_theta, ndim, n]
        if points.is_cuda:
            gradient = cpab_gpu.backward(points.contiguous(), 
                                         As.contiguous(), 
                                         Bs.contiguous(), 
                                         nstepsolver.contiguous(), 
                                         nc.contiguous())
        else:
            gradient = cpab_cpu.backward(points.contiguous(), 
                                         As.contiguous(), 
                                         Bs.contiguous(), 
                                         nstepsolver.contiguous(), 
                                         nc.contiguous())
            
        # Backpropagate and reduce to [d, n_theta] matrix
        g = gradient.mul_(grad).sum(dim=(2,3))
        return None, g.t() # transpose, since pytorch expects a [n_theta, d] matrix

#%%
class _CPABFunction_numeric(torch.autograd.Function):
    # Function that connects the forward pass to the backward pass
    @staticmethod
    def forward(ctx, points, theta):
        assert points.device == theta.device, '''points are on device %s and 
            theta are on device %s. Please make sure that they are on the 
            same device''' % (str(points.device), str(theta.device))
        device = points.device
        params = load_basis_as_struct()
        
        # Problem size
        n_theta = theta.shape[0]
        
        # Get velocity fields
        B = torch.Tensor(params.basis).to(device)
        Avees = torch.matmul(B, theta.t())
        As = Avees.t().reshape(n_theta*params.nC, *params.Ashape)
        zero_row = torch.zeros(n_theta*params.nC, 1, params.ndim+1).to(device)
        AsSquare = torch.cat([As, zero_row], dim=1)
        
        # Take matrix exponential
        dT = 1.0 / params.nstepsolver
        Trels = torch_expm(dT*AsSquare)
        Trels = Trels[:,:params.ndim,:].view(n_theta, params.nC, *params.Ashape)
        
        # Convert to tensor
        nstepsolver = torch.tensor(params.nstepsolver, dtype=torch.int32, 
                                   device=device).to(device)
        nc = torch.tensor(params.nc, dtype=torch.int32, device=device)
        
        # Call integrator
        if points.is_cuda:
            newpoints = cpab_gpu.forward(points.contiguous(), 
                                         Trels.contiguous(), 
    								nstepsolver.contiguous(), 
												nc.contiguous())
        else:            
            newpoints = cpab_cpu.forward(points.contiguous(), 
												Trels.contiguous(), 
												nstepsolver.contiguous(), 
												nc.contiguous())
        
        # Save for backward
        ctx.save_for_backward(points, theta, newpoints, nstepsolver, nc, params)
        # Output result
        return newpoints
    
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad): # grad [n_theta, ndim, n]
        # Grap input
        points, theta, newpoints, nstepsolver, nc, params = ctx.saved_tensors
        device = points.device
        h = 0.01
        gradient = [ ]
        
        # Problem size
        n_theta, d = theta.shape
        
        for i in range(d):
            # Permute theta
            temp = theta
            temp[i] += h
            
            # Get velocity fields
            B = torch.Tensor(params.basis).to(device)
            Avees = torch.matmul(B, temp.t())
            As = Avees.t().reshape(n_theta*params.nC, *params.Ashape)
            zero_row = torch.zeros(n_theta*params.nC, 1, params.ndim+1).to(device)
            AsSquare = torch.cat([As, zero_row], dim=1)
            
            # Take matrix exponential
            dT = 1.0 / params.nstepsolver
            Trels = torch_expm(dT*AsSquare)
            Trels = Trels[:,:params.ndim,:].view(n_theta, params.nC, *params.Ashape)
            
            # Call integrator
            if points.is_cuda:
                temp_points = cpab_gpu.forward(points.contiguous(), 
                                             Trels.contiguous(), 
        								nstepsolver.contiguous(), 
    												nc.contiguous())
            else:            
                temp_points = cpab_cpu.forward(points.contiguous(), 
    												Trels.contiguous(), 
    												nstepsolver.contiguous(), 
    												nc.contiguous())
            
            diff = (temp_points - newpoints) / h
            
            # Do finite gradient
            gradient.append((grad * diff).sum(dim=[1,2])) # gradient [n_theta, ]
        
        # Reshaping
        gradient = torch.stack(gradient, dim = 1) # [n_theta, d]
        return gradient

#%%
def _fast_transformer(points, theta):
    return _CPABFunction.apply(points, theta)

#%%
def _fast_transformer_numeric(points, theta):
    return _CPABFunction_numeric.apply(points, theta)
.is_cuda: res = expm_gpu_module.forward(A.contiguous())
                else: r
#%%
def _slow_transformer(points, theta):
    assert points.device == theta.device, '''points are on device %s and 
            theta are on device %s. Please make sure that they are on the 
            same device''' % (str(points.device), str(theta.device))
    device = points.device
    params = load_basis_as_struct()
    
    # Problem sizes
    n_points = points.shape[1]
    n_theta = theta.shape[0]
    
    # Transform points
    ones = torch.ones(n_theta, 1, n_points).to(device)
    newpoints = torch_repeat_matrix(points, n_theta) # [n_theta, ndim, n_points]
    newpoints = torch.cat([newpoints, ones], dim=1)
    newpoints = newpoints.permute(0, 2, 1).reshape(-1, params.ndim+1)
    newpoints = newpoints[:,:,None] # [n_theta * ndim, ndim+1, 1]
        
    # Get velocity fields
    B = torch.Tensor(params.basis).to(device)
    Avees = torch.matmul(B, theta.t())
    As = Avees.t().reshape(n_theta*params.nC, *params.Ashape)
    zero_row = torch.zeros(n_theta*params.nC, 1, params.ndim+1).to(device)
    AsSquare = torch.cat([As, zero_row], dim=1)
        
    # Take matrix exponential
    dT = 1.0 / params.nstepsolver
    Trels = torch_expm(dT*AsSquare)

    # Batch index to add to correct for the batch effect
    batch_idx = params.nC * (torch.ones(n_points, n_theta, dtype=torch.int64) * 
                                  torch.arange(n_theta)).t().flatten()
    batch_idx = batch_idx.to(device)
    
    # Intergrate velocity field
    for i in range(params.nstepsolver):
        # Find cell index and correct for batching effect
        idx = torch_findcellidx(params.ndim, newpoints, params.nc) + batch_idx
        
        # Gather the correct transformations
        Tidx = Trels[idx]
        
        # Transform points
        newpoints = torch.matmul(Tidx, newpoints)
        del idx, Tidx
    
    # Reshape to the right format
    newpoints = newpoints.squeeze()[:,:params.ndim].t()
    newpoints = newpoints.reshape(params.ndim, n_theta, n_points).permute(1,0,2)
    return newpoints
    
#%%
def CPAB_transformer(points, theta):
    ''' This wrapper function, combines the fast and slow transformer into one
        function. If the fast versions have been successfully compiled, then these
        will be used, else default to the slow versions '''
    if points.is_cuda and theta.is_cuda:
        if _gpu_succes and not _use_slow:
            if _verbose: print('fast gpu version')
            if _use_numeric:
                if _verbose: print('using finite difference')
                newpoints = _fast_transformer_numeric(points, theta)    
            else:
                newpoints = _fast_transformer(points, theta)
        else:
            if _verbose: print('slow gpu version')
            newpoints = _slow_transformer(points, theta)
    else:
        if _cpu_succes and not _use_slow:
            if _verbose: print('fast cpu version')
            if _use_numeric:
                if _verbose: print('using finite difference')
                newpoints = _fast_transformer_numeric(points, theta)
            else:
                newpoints = _fast_transformer(points, theta)
        else:
            if _verbose: print('slow cpu version')
            newpoints = _slow_transformer(points, theta)
    return newpoints       

#%%
if __name__ == '__main__':
    pass
