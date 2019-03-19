#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:26:03 2019

@author: nsde
"""
from torch.utils.cpp_extension import load
import os
def get_dir(file):
    """ Get directory of the input file """
    return os.path.dirname(os.path.realpath(file))

_dir = get_dir(__file__)
cpab_cpu = load(name = 'cpab_cpu',
                sources = [_dir + '/transformer.cpp',
                           _dir + '/../core/cpab_ops.cpp'],
                verbose=True)

cpab_gpu = load(name = 'cpab_gpu',
                sources = [_dir + '/transformer_cuda.cpp',
                           _dir + '/transformer_cuda.cu',
                           _dir + '/../core/cpab_ops.cu'],
                verbose=True,
                with_cuda=True)