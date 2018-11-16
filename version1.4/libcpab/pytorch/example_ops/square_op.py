# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 08:41:19 2018

@author: nsde
"""



#%%
import torch
from torch.utils.cpp_extension import load
import warnings

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
try:
    with warnings.catch_warnings(record=True) as w:
        square_cpu = load(name = 'square_cpu',
                          sources = ['cpp_extension/square.cpp'])
    cpu_succes = True
except:
    square_cpu = _notcompiled()
    cpu_succes = False

# Compile gpu source
try:
    with warnings.catch_warnings(record=True) as w:
        square_gpu = load(name = 'square_cuda',
                          sources = ['cuda_extension/square_cuda.cpp', 
                                     'cuda_extension/square_cuda_kernel.cu'], 
                          verbose=False)
    gpu_succes = True
except:
    square_gpu = _notcompiled()
    gpu_succes = False
    
#%%
class _squareFunction(torch.autograd.Function):
    # Function that connects the forward pass to the backward pass
    @staticmethod
    def forward(ctx, input):
        if input.is_cuda:
            output = square_gpu.forward(input)
        else:
            output = square_cpu.forward(input)
        ctx.save_for_backward(input)
        return output
        
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        if input.is_cuda:
            output = square_gpu.backward(input)
        else:
            output = square_cpu.backward(input)
        return output * grad_in

#%%
def _square_fast(input):
    # Wrapper function that applies the operation
    return _squareFunction.apply(input)

#%%
def _square_slow(input):
    # Analytic solution we can default to 
    return input.pow(2.0)

#%%
def square(input):
    # Combined solution, that will call the fast version if we are able to compile it
    if input.is_cuda:
        if gpu_succes:
            output = _square_fast(input)
        else:
            output = _square_slow(input)
    else:
        if cpu_succes:
            output = _square_fast(input)
        else:
            output = _square_slow(input)
    return output        
    
#%%
if __name__ == '__main__':
    import numpy as np
    x = np.random.normal(size=(2,2))
    print('Input')
    print(x)

    print('\nCPU test')
    x1 = torch.autograd.Variable(torch.Tensor(x), requires_grad=True)
    x2 = torch.autograd.Variable(torch.Tensor(x), requires_grad=True)
    print('forward pass')
    print(square(x1))
    print(x2.pow(2))
    print('backward pass')
    print(torch.autograd.grad(square(x1).sum(), x1))
    print(torch.autograd.grad(x2.pow(2).sum(), x2))
    
    print('\nGPU test')
    x1 = torch.autograd.Variable(torch.Tensor(x), requires_grad=True).cuda()
    x2 = torch.autograd.Variable(torch.Tensor(x), requires_grad=True).cuda()
    print('forward pass')
    print(square(x1))
    print(x2.pow(2))
    print('backward pass')
    print(torch.autograd.grad(square(x1).sum(), x1))
    print(torch.autograd.grad(x2.pow(2).sum(), x2))
    
