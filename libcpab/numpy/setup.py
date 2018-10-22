#!/usr/bin/env python3
# encoding: utf-8

from distutils.core import setup, Extension

nplibcpab_module = Extension('nplibcpab', sources = ['src/npcpab1.cc'], extra_compile_args=["-Ofast", "-march=native"])

setup(name             = "libcpab_numpy_backend",
      version          = "1.0",
      description      = "libcpab numpy backend; not to be called directly",
      author           = "Soren Hauberg",
      author_email     = "sohau@dtu.dk",
      maintainer       = "sohau@dtu.dk",
      url              = "https://github.com/SkafteNicki/libcpab",
      ext_modules      = [nplibcpab_module]
)
