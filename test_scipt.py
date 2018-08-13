# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:50:53 2018

@author: nsde
"""
if __name__ == '__main__':
    from libcpab import cpab
    T = cpab(tess_size=[3,3], return_tf_tensors=True)
    g = T.uniform_meshgrid([20,20])
    theta = T.sample_transformation(1)
    gt = T.transform_grid(g, theta)
    #v=T.calc_vectorfield(g, theta[0])
    #T.visualize_vectorfield(theta[0], 15)
    
