#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:51:31 2019

@author: nsde
"""

#%%
from libcpab import Cpab
from libcpab import CpabAligner

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
    parser.add_argument('--alignment_type', default='sampling',
                        choices=['sampling', 'gradient'],
                        help='how to align samples')
    parser.add_argument('--maxiter', type=int, default=100,
                        help='number of iteration in alignment algorithm')
    return parser.parse_args()


#%%
if __name__ == "__main__":
    args = argparser()

    # Load some data
    data = plt.imread('cat.jpg') / 255
    data = np.expand_dims(data, 0)  # create batch effect

    # Create transformer class
    T = Cpab([3, 3], backend=args.backend, device=args.device, zero_boundary=True,
             volume_perservation=False, override=False)

    # Sample random transformation
    theta = T.sample_transformation(1)

    # Convert data to the backend format
    data = T.backend.to(data, device=args.device)

    # Pytorch have other data format than tensorflow and numpy, color information
    # is the second dim. We need to correct this before and after
    data = data.permute(0, 2, 3, 1) if args.backend == 'pytorch' else data

    # Transform the images
    transformed_data = T.transform_data(data, theta, outsize=(350, 350))

    # Now lets see if we can esimate the transformation we just used, by
    # iteratively trying to transform the data
    A = CpabAligner(T)

    # Do by sampling, work for all backends
    if args.alignment_type == 'sampling':
        theta_est = A.alignment_by_sampling(
            data, transformed_data, maxiter=args.maxiter)
    else:
        theta_est = A.alignment_by_gradient(
            data, transformed_data, maxiter=args.maxiter)

    # Lets see what we converged to
    trans_est = T.transform_data(data, theta_est, outsize=(350, 350))
    trans_est = trans_est.permute(
        0, 2, 3, 1) if args.backend == 'pytorch' else trans_est

    # Show the results
    data = T.backend.tonumpy(data)
    transformed_data = T.backend.tonumpy(transformed_data)
    trans_est = T.backend.to(trans_est)

    plt.subplots(1, 3)
    plt.subplot(1, 3, 1)
    plt.imshow(data[0])
    plt.axis('off')
    plt.title('Source')
    plt.subplot(1, 3, 2)
    plt.imshow(transformed_data[0])
    plt.axis('off')
    plt.title('Target')
    plt.subplot(1, 3, 3)
    plt.imshow(trans_est[0])
    plt.axis('off')
    plt.title('Estimate')
    plt.show()
