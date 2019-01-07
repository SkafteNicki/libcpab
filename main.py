# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:40:59 2018

@author: nsde
"""

if __name__ == "__main__":
    import torch    
    from libcpab import cpab, DataAligner, SequentialCpab
    T = cpab([2,2], backend='pytorch')
    theta = T.sample_transformation_with_prior(1)
    