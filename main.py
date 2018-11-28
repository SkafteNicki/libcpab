# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:40:59 2018

@author: nsde
"""

if __name__ == "__main__":
    from libcpab import cpab
    import numpy as np
    
    # 1D
    T1 = cpab([5,], backend="numpy")
    B1 = T1.get_basis()
    P1 = T1.get_params()
    d1 = T1.get_theta_dim()
    g1 = T1.uniform_meshgrid([10])
    t1 = T1.identity()
    s1 = T1.sample_transformation()
    s1 = T1.sample_transformation(1, np.zeros(d1), np.eye(d1))
    v1 = T1.calc_vectorfield(g1, s1)
    T1.visualize_vectorfield(s1)
    T1.visualize_tesselation()
    #gt1 = T1.transform_grid(g1, s1)
    #D1 = T1.transform_data()
    
    # 2D
    T2 = cpab([2,2], backend="numpy")
    B2 = T2.get_basis()
    P2 = T2.get_params()
    d2 = T2.get_theta_dim()
    g2 = T2.uniform_meshgrid([10, 10])
    t2 = T2.identity()
    s2 = T2.sample_transformation()
    s2 = T1.sample_transformation(1, np.zeros(d2), np.eye(d2))
    #v2 = T2.calc_vectorfield(g2, s2)
    #T2.visualize_vectorfield(s2)
    #T2.visualize_tesselation()
    
    # 3D
    T3 = cpab([2,2,2], backend="numpy")
    B3 = T3.get_basis()
    P3 = T3.get_params()
    d3 = T3.get_theta_dim()
    g3 = T3.uniform_meshgrid([10, 10, 10])
    t3 = T3.identity()
    s3 = T3.sample_transformation()
    s3 = T1.sample_transformation(1, np.zeros(d3), np.eye(d3))
    #v3 = T3.calc_vectorfield()
    #T3.visualize_vectorfield(s1)
    #T3.visualize_tesselation()