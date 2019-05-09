#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:29:44 2019

@author: nsde
"""

#%%
import unittest
from libcpab import cpab

#%%
class Test(unittest.TestCase):
    
    tess_sizes = [(2,), (3,), (4,), (5,), (1,1), (2,2), (3,3), (4,4)]
    backends = ['numpy', 'pytorch']
    theta_dim = [34]
    
    
    def test_theta_dim(self):
        for b in self.backends:
            T = cpab([3,3], backend=b, device='cpu', zero_boundary=True, 
                     volume_perservation=False, override=False)
            self.assertEqual(T.get_theta_dim(), 34, 'Backend ' + b + ' failed')
            
    def test_grid_size(self):
        pass
        
    def test_theta_size(self):
        pass
    
#%%
if __name__ == '__main__':
    unittest.main()