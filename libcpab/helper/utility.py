#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:39:44 2017

@author: nsde
"""
#%%
try:
    import cPickle as pkl
except:
    import pickle as pkl
import os, random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.client import device_lib 
from sys import platform as _platform

#%%
def gpu_support():
    gpu = check_for_gpu() and check_cuda_support()
    if (_platform == "linux" or _platform == "linux2" \
        or _platform == "darwin") and gpu: # linux or MAC OS X
        return True
    else: # Windows 32 or 64-bit or no GPU
        return False

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
def make_hashable(arr):
    """ Make an array hasable. In this way we can use built-in functions like
        set(...) and intersection(...) on the array
    """
    return tuple([tuple(r.tolist()) for r in arr])

#%%
def load_obj(name):
    """ Function for saving a variable as a pickle file """
    with open(name + '.pkl', 'rb') as f:
        return pkl.load(f)

#%%
def save_obj(obj, name):
    """ Function for loading a pickle file """
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

#%%
def get_path(file):
    """ Get the path of the input file """
    return os.path.realpath(file)

#%%
def get_dir(file):
    """ Get directory of the input file """
    return os.path.dirname(os.path.realpath(file))

#%%
def create_dir(direc):
    """ Create a dir if it does not already exists """
    if not os.path.exists(direc):
        os.mkdir(direc)

#%%
def check_if_file_exist(file):
    return os.path.isfile(file)

#%%
def load_basis():
    """ Load the latest cpab basis created by cpab class """
    basis_loc = get_dir(__file__) + '/../basis_files/current_basis'
    try:
        basis = load_obj(basis_loc)
    except:
        raise ValueError('call the cpab class first to create basis')
    return basis

#%%
def uniqueid_generator(x):
    """ Function to generate uniquely id of x bits """
    seed = random.getrandbits(x)
    while True:
       yield str(seed)
       seed += 1
uniqueid = uniqueid_generator(12)

#%%
def show_images(images, cols='auto', title=None, scaling=False):
    """ Display a list of images in a single figure with matplotlib.
    
    Arguments
        images: List/tensor of np.arrays compatible with plt.imshow.
    
        cols (Default = 'auto'): Number of columns in figure (number of rows is 
                                 set to np.ceil(n_images/float(cols))).
        
        title: One main title for the hole figure
            
        scaling (Default = False): If True, will rescale the figure by the
                number of images. Good if one want to show many.
    """
    n_images = len(images)
    cols = np.round(np.sqrt(n_images)) if cols=='auto' else cols
    rows = np.ceil(n_images/float(cols))
    fig = plt.figure()
    if type(title)==str: fig.suptitle(title, fontsize=20)
    for n, image in enumerate(images):
        a = fig.add_subplot(cols, rows, n + 1)
        if image.ndim == 2: plt.gray()
        a.imshow(image)
        a.axis('on')
        a.axis('equal')
        a.set_xticklabels([])
        a.set_yticklabels([])
    if scaling: fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()

#%%
if __name__ == '__main__':
    pass    