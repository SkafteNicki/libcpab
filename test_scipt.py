# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:50:53 2018

@author: nsde
"""
if __name__ == '__main__':
    from libcpab import cpab
    T = cpab(tess_size=[2,2,2])
    g = T.sample_grid([5,5,5])
    theta = T.sample_transformation(2)
    gt = T.transform(g, theta)
    
