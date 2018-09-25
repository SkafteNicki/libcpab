# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 09:54:53 2018

@author: nsde
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='square_gpu',
    ext_modules=[CUDAExtension('square_gpu', ['square_cuda.cpp',
                                              'square_cuda_kernel.cu'])],
    cmdclass={'build_ext': BuildExtension}
    )
