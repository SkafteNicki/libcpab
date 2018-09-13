#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 13:55:41 2018

@author: nsde
"""
#%%
import tensorflow as tf
from tensorflow.python.client import device_lib 
from sys import platform as _platform

#%%
def check_for_gpu():
    devices = device_lib.list_local_devices()
    gpu = False
    for d in devices:
        if d.device_type == "GPU": gpu=True
    return gpu

#%%
def check_cuda_support():
    return tf.test.is_built_with_cuda()

#%%
def gpu_support():
    gpu = check_for_gpu() and check_cuda_support()
    if (_platform == "linux" or _platform == "linux2") and gpu: # linux or MAC OS X
        return True
    else: # Windows 32 or 64-bit or no GPU
        return False
