#!/usr/bin/env python3
# encoding: utf-8

from distutils.core import setup, Extension
import sys

extra_compile_args = []
extra_link_args = []
if sys.platform.startswith('win'):
    extra_compile_args.append('-openmp')
else:
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-lgomp')

nplibcpab_module = Extension('nplibcpab', sources = ['src/npcpab1.cc'], extra_compile_args=extra_compile_args, extra_link_args=extra_link_args)

setup(name             = "libcpab_numpy_backend",
      version          = "1.0",
      description      = "libcpab numpy backend; not to be called directly",
      author           = "Soren Hauberg",
      author_email     = "sohau@dtu.dk",
      maintainer       = "sohau@dtu.dk",
      url              = "https://github.com/SkafteNicki/libcpab",
      ext_modules      = [nplibcpab_module]
)
