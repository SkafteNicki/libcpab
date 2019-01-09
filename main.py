# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:40:59 2018

@author: nsde
"""

if __name__ == "__main__":
    import torch    
    from libcpab import cpab
    T = cpab([3,3], backend='pytorch', device='gpu', zero_boundary=True, 
             volume_perservation=False, override=False)
    