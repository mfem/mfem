#!/bin/bash

ml cuda/13.1.1
ml clang/19.1.3-magic

make cuda CUDA_ARCH=sm_100 -j100
