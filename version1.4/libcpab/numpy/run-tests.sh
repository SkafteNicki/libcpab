#!/bin/bash

rm -rf build
python3 setup.py build
cd build/lib.linux-x86_64-3.6/
cp ../../run-tests.py .
python3 run-tests.py
cd ../..
