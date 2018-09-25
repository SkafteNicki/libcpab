# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 08:41:19 2018

@author: nsde
"""

#%%
import torch
import square as square_ops

#%%
class squareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = square_ops.forward(input)
        ctx.save_for_backward(input)
        return output
        
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        print(input)
        output = square_ops.backward(input)
        return output * grad_in

#%%
def square(input):
    return squareFunction.apply(input)

#%%
if __name__ == '__main__':
    import numpy as np
    x = np.random.normal(size=(2,2))
    print(x)
    x1 = torch.autograd.Variable(torch.Tensor(x), requires_grad=True)
    x2 = torch.autograd.Variable(torch.Tensor(x), requires_grad=True)
    print(torch.autograd.grad(square(x1).sum(), x1))
    print(torch.autograd.grad(x2.pow(2).sum(), x2))
    
    
