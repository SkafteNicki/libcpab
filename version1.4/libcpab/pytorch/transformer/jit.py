# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 07:59:09 2018

@author: nsde
"""
#%%
import torch
from torch.utils.cpp_extension import load

#%%
if __name__ == '__main__':
#   _dir = get_dir(__file__)  
    
    # Compile cpu source
    cpab_cpu = load(name = 'cpab_cpu',
                sources = ['CPAB_ops.cpp'],
                verbose=True)
    
    # Compule gpu source
    cpab_gpu = load(name = 'cpab_gpu',
                sources = ['CPAB_ops_cuda.cpp', 
                           'CPAB_ops_cuda_kernel.cu'],
                verbose=True)
