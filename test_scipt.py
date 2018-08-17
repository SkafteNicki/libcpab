# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:50:53 2018

@author: nsde
"""
if __name__ == '__main__':
    from libcpab import cpab
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    data = plt.imread('data/cat.jpg')
    data = np.expand_dims(cv2.resize(data, (300, 300)),0)/255
    T = cpab(tess_size=[3,3])
    theta = T.sample_transformation(1)
    newim = T.transform_data(data, theta)
    plt.imshow(newim[0])
    #v=T.calc_vectorfield(g, theta[0])
    #T.visualize_vectorfield(theta[0], 15)
    
    
    
    # TODO: messing up the dimensions of the interpolation
    # TODO: fix_data_size incorporate compiled versions
    