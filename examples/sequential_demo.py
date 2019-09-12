#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:05:57 2019

@author: nsde
"""

#%%
from libcpab import Cpab
from libcpab import CpabSequential
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
if __name__ == '__main__':
    args = argparser()
    print("---Running script with arguments---")
    print("\n".join([str(k) + ':' + str(v) for k,v in vars(args).items()]))
    print("-----------------------------------")
    
    # Load some data
    data = plt.imread(get_dir(__file__) + '/../data/cat.jpg') / 255
    data = np.expand_dims(data, 0)  # create batch effect

    # Create multiple transformers
    T1 = Cpab([1, 1], backend=args.backend, device=args.device, zero_boundary=True,
              volume_perservation=False, override=False)
    T2 = Cpab([2, 2], backend=args.backend, device=args.device, zero_boundary=True,
              volume_perservation=False, override=False)
    T3 = Cpab([3, 3], backend=args.backend, device=args.device, zero_boundary=True,
              volume_perservation=False, override=False)

    # Combine into one single transformer
    T = CpabSequential(T1, T2, T3)

    # Sample transformations (one for each transformation)
    thetas = T.sample_transformation(1)
    # scale down, else the output will be very deform
    thetas = [T.backend.to(0.5*t, device=args.device) for t in thetas]

    # Convert data to the backend format
    data = T.backend.to(data, device=args.device)

    # Pytorch have other data format than tensorflow and numpy, color information
    # is the second dim. We need to correct this before and after
    data = data.permute(0, 3, 1, 2) if args.backend == 'pytorch' else data

    # Transform the images, output_all=True will return all intermidian transforms
    all_transformed_data = T.transform_data(data, 
                                            thetas, 
                                            outsize=(350, 350), 
                                            output_all=True)

    # Get the corresponding numpy arrays in correct format
    transformed_data = [data.permute(0, 2, 3, 1) if args.backend == 'pytorch' else data]
    transformed_data[0] = T.backend.tonumpy(transformed_data[0])

    for td in all_transformed_data:
        td = td.permute(0, 2, 3, 1) if args.backend == 'pytorch' else td
        transformed_data.append(T.backend.tonumpy(td))
    transformed_data = np.concatenate(transformed_data, axis=0)

    # Show transformed samples
    show_images(transformed_data)
