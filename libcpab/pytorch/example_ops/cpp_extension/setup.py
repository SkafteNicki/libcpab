# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 17:39:44 2018

@author: nsde
"""

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='square_cpu',
      ext_modules=[CppExtension('square_cpu', ['square.cpp'])],
      cmdclass={'build_ext': BuildExtension})
