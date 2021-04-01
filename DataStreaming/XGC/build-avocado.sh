#!/bin/bash

mkdir avocado-build
rm -r avocado-build/*
cd avocado-build

module load gcc/9.3.0
module load spectrum-mpi
module load cmake
module load python/3.7.0

effis=/gpfs/alpine/csc143/world-shared/esuchyta/spack/opt/spack/linux-rhel7-power9le/gcc-9.3.0/effis-develop-trvkjzyfbsh3d3zfds66h3wahd7ylj4o
vtkm=/gpfs/alpine/csc143/world-shared/esuchyta/spack/opt/spack/linux-rhel7-power9le/gcc-9.3.0/vtk-m-master-672m33oblm5xiazctojrd4a3v4vowk33
cuda=/sw/summit/cuda/11.1.0
#adios=/gpfs/alpine/csc143/world-shared/esuchyta/spack/opt/spack/linux-rhel7-power9le/gcc-9.3.0/adios2-master-3763asrgriktrvq2suiyqrksg6svooce
adios=/ccs/home/againaru/adios/ADIOS2-init/install

cmake \
	-DCMAKE_VERBOSE_MAKEFILE=ON \
	-DCMAKE_PREFIX_PATH="$effis;$vtkm;$cuda;$adios" \
	-DEFFIS=ON \
	../vtk-avocado

make
