#!/bin/bash
EXTRA_ARGS=("$@")
rm -f  CMakeCache.txt
rm -rf CMakeFiles/
ARGS=(
    -D CMAKE_BUILD_TYPE=RelWithDebInfo
    -D CMAKE_INSTALL_PREFIX=$HOME/kokkos/install
    -D CMAKE_CXX_COMPILER=$HOME/kokkos/kokkos/bin/nvcc_wrapper

    -D Kokkos_ENABLE_SERIAL=ON
    -D Kokkos_ARCH_POWER9=ON

    -D Kokkos_ENABLE_CUDA=ON
    -D Kokkos_ENABLE_CUDA_LAMBDA=ON
    -D Kokkos_ARCH_VOLTA70=ON

    -D CMAKE_CXX_STANDARD=17
    -D CMAKE_CXX_EXTENSIONS=OFF

    -D CMAKE_POSITION_INDEPENDENT_CODE=TRUE
)
cmake "${ARGS[@]}" "${EXTRA_ARGS[@]}" ../kokkos
