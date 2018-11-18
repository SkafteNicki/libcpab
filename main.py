# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:40:59 2018

@author: nsde
"""

if __name__ == "__main__":
    from libcpab import cpab
    T1_numpy = cpab([2,], backend="numpy")
    T2_numpy = cpab([2,3], backend="numpy")
    T3_numpy = cpab([2,3,4], backend="numpy")
    
    T1_pytorch = cpab([2,], backend="pytorch")
    T2_pytorch = cpab([2,3], backend="pytorch")
    T3_pytorch = cpab([2,3,4], backend="pytorch")
    
    T1_tensorflow = cpab([2,], backend="tensorflow")
    T2_tensorflow = cpab([2,3], backend="tensorflow")
    T3_tensorflow = cpab([2,3,4], backend="tensorflow")
