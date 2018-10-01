# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 07:59:09 2018

@author: nsde
"""
#%%
from torch.utils.cpp_extension import load
from ...helper.utility import get_dir

#%%
if __name__ == '__main__':
    _dir = get_dir(__file__)  
    
    # Compile cpu source
    cpab_cpu = load(name = 'cpab_cpu',
                sources = [_dir + '/CPAB_ops.cpp'],
                verbose=True)
    
    # Compule gpu source
    cpab_gpu = load(name = 'cpab_gpu',
                sources = [_dir + '/CPAB_ops_cuda.cpp', 
                           _dir + '/CPAB_ops_cuda_kernel.cu'],
                verbose=True)
