# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:29:26 2019

@author: nsde
"""

#%%
from libcpab import Cpab
from libcpab.core.utility import show_images, get_dir # utility functions

import numpy as np
import matplotlib.pyplot as plt
import argparse

#%%
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='numpy', 
                        choices=['numpy', 'tensorflow', 'pytorch'],
                        help='backend to run demo with')
    parser.add_argument('--device', default='cpu',
                        choices=['cpu', 'gpu'],
                        help='device to run demo on')
    return parser.parse_args()

#%%
if __name__ == "__main__":
    args = argparser()
    print("---Running script with arguments---")
    print("\n".join([str(k) + ':' + str(v) for k,v in vars(args).items()]))
    print("-----------------------------------")
    
    import tensorflow as tf
    tf.config.set_soft_device_placement(True)
    
    # Number of transformed samples 
    N = 9
    
    # Load some data
    data = plt.imread(get_dir(__file__) + '/../data/cat.jpg') / 255
    data = np.tile(data[None], [N,1,1,1]) # create batch of data
    
    # Create transformer class
    T = Cpab([3, 3], backend=args.backend, device=args.device, 
             zero_boundary=True, volume_perservation=False, override=False)

    # Sample random transformation
    theta = T.sample_transformation(N)
    
    # Convert data to the backend format
    data = T.backend.to(data, device=args.device)
    
    # Pytorch have other data format than tensorflow and numpy, color 
    # information is the second dim. We need to correct this before and after
    data = data.permute(0,3,1,2) if args.backend=='pytorch' else data

    # Transform the images
    t_data = T.transform_data(data, theta, outsize=(350, 350))
    
    # Get the corresponding numpy arrays in correct format
    t_data = t_data.permute(0,2,3,1) if args.backend=='pytorch' else t_data
    t_data = T.backend.tonumpy(t_data)
    
    # Show transformed samples
    show_images(t_data)
