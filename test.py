# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 08:09:00 2018

@author: nsde
"""

from libcpab.pytorch import cpab

device = 'cpu'
T = cpab(tess_size=[2,2,2], device=device)
g = T.uniform_meshgrid([200, 200, 200])
theta = 0.1*T.sample_transformation(1)
gt = T.transform_grid(g, theta)