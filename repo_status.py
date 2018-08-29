# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:50:53 2018

@author: nsde
"""

#%%
# Future repo structure
#   -libcpab/
#       *tensorflow/
#           +cpab1d/
#               -...
#           +cpab2d/
#               -...
#           +cpab3d/
#               -...
#           +helper/
#               -...
#           +__init__.py
#           +cpab.py
#       *pytorch/
#           +__init__.py
#           +cpab.py
#           +helper.py
#       *helper/
#           +utility.py
#           +math.py
#       *basis_files/
#           +...
#   -data/
#       *...
#   -demo1.py
#   -demo2.py
#   -demo3.py
#
# Future import structure
# from libcpab import tf_cpab as cpab
# or
# from libcpab import torch_cpab as cpab
#

#%%
if __name__ == '__main__':
    from libcpab.develop.cpab import cpab
