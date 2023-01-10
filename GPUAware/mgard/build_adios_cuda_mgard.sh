#!/bin/sh

# Copyright 2022, Oak Ridge National Laboratory.
# ADIOS2: The Adaptable Input Output System version 2
# Author: Ana Gainaru (gainarua@ornl.gov)
# Date: January 10, 2022
# Script for building ADIOS2 on Summit

set -e
set -x

module load cuda/11.4
module load gcc/9
module load cmake

adios2_src_dir=/path/to/adios/ADIOS2
adios2_build_dir=${adios2_src_dir}/build
adios2_install_dir=${adios2_src_dir}/install
mgard_install_dir=/path/to/MGARD/install-cuda-summit

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${mgard_install_dir}/lib64:${mgard_install_dir}/lib

mkdir -p ${adios2_install_dir}

cmake -S ${adios2_src_dir} -B ${adios2_build_dir} \
    -DADIOS2_USE_CUDA=ON\
    -DADIOS2_USE_MGARD=ON\
	-DMGARD_ROOT=${adios2_install_dir}\
	-DADIOS2_USE_SST=OFF\
    -DADIOS2_BUILD_EXAMPLES=OFF\
    -DCMAKE_INSTALL_PREFIX=${adios2_install_dir}\
	-DBUILD_TESTING=OFF\
	-DCMAKE_C_COMPILER=gcc\
	-DCMAKE_CXX_COMPILER=g++\
    -DCMAKE_CUDA_ARCHITECTURES="70"

make -j4
make -j4 install
