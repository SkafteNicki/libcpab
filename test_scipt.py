# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:50:53 2018

@author: nsde
"""
if __name__ == '__main__':
    from libcpab import cpab
    T = cpab(tess_size=[2,2,2])
    g = T.sample_grid([20,20,20])
    theta = T.sample_transformation(1) 
    #gt = T.transform(g, theta)
    v=T.calc_v(g, theta[0])
    T.visualize_vector_field(0.1*theta[0], 15)
    
