#!/bin/bash

mkdir xgc-build
rm -r xgc-build/*
cd xgc-build

module load pgi/19.10
module load pgi-cxx14
module load spectrum-mpi
module load cmake
module load cuda/10.1.243

effis=/gpfs/alpine/csc143/world-shared/esuchyta/spack/opt/spack/linux-rhel7-power9le/pgi-19.10/effis-develop-rgzjnggufdzpus7pqco2wza3hjzlyxty
export XGC_PLATFORM=summit
export NVCC_WRAPPER_DEFAULT_COMPILER=pgc++

cmake \
	-DCMAKE_CXX_COMPILER=/gpfs/alpine/world-shared/phy122/lib/install/summit/kokkos/pgi19.10/bin/nvcc_wrapper \
	-DCMAKE_Fortran_COMPILER=mpif90 \
	-DXGC_USE_CABANA=ON \
	-DBUILD_TESTING=OFF \
	-DBUILD_KERNELS=OFF \
	-DEFFIS=ON \
	-DCMAKE_PREFIX_PATH=$effis \
	../XGC-Devel

make xgc-es-cpp-gpu
