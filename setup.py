# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 15:26:46 2018

@author: nsde
"""

#%%
from distutils.core import setup

#%%
setup(name = "libcpab",
      version = "2.0",
      description = "Diffiomorphism for dummies",
      author = "Nicki Skafte Detlefsen",
      author_email = "nsde@dtu.dk",
      packages = ["libcpab"],
      license = "MIT",
      long_description = open('README.md').read())