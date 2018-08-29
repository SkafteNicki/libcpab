# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 11:35:24 2018

@author: nsde
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='CPAB_ops',
    ext_modules=[
        CUDAExtension('CPAB_ops', [
            'CPAB_ops.cpp',
            'CPAB_ops.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })