# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 08:09:00 2018

@author: nsde
"""

from libcpab.pytorch import cpab
import time

device = 'cpu'
T = cpab(tess_size=[2,2,2], device=device)
g = T.uniform_meshgrid([100, 100, 100])
theta = 0.1*T.sample_transformation(1)
#theta.requires_grad = False
start = time.time()
gt = T.transform_grid(g, theta)
print(time.time() - start)
