# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 10:55:35 2018

@author: nsde
"""

# Run both setup files
import os
os.system('python cpp_extension/setup.py install')
os.system('python cuda_extension/setup.py install')