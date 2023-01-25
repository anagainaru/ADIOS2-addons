module load cuda/11.4 gcc/10.2.0 cmake

KOKKOS_HOME=/path/to/kokkos

git clone git@github.com:kokkos/kokkos.git ${KOKKOS_HOME}/kokkos
git checkout develop
mkdir -p ${KOKKOS_HOME}/install
mkdir -p ${KOKKOS_HOME}/build

cd ${KOKKOS_HOME}/build

ARGS=(
    -D CMAKE_BUILD_TYPE=RelWithDebInfo
    -D CMAKE_INSTALL_PREFIX=${KOKKOS_HOME}/install
    -D CMAKE_CXX_COMPILER=${KOKKOS_HOME}/kokkos/bin/nvcc_wrapper

    -D Kokkos_ENABLE_SERIAL=ON
    -D Kokkos_ARCH_POWER9=ON

    -D Kokkos_ENABLE_CUDA=ON
    -D Kokkos_ENABLE_CUDA_LAMBDA=ON
    -D Kokkos_ARCH_VOLTA70=ON

    -D CMAKE_CXX_STANDARD=17
    -D CMAKE_CXX_EXTENSIONS=OFF

    -D CMAKE_POSITION_INDEPENDENT_CODE=TRUE
)
cmake "${ARGS[@]}" ../kokkos
make -j4
make install

ADIOS2_HOME=/path/to/adios2

cd ${ADIOS2_HOME}
git clone git@github.com:ornladios/ADIOS2.git
mkdir -p build
mkdir -p install

ARGS_ADIOS@=(
     -D CMAKE_CXX_STANDARD=17 
     -D CMAKE_CXX_EXTENSIONS=OFF  
     -DCMAKE_C_COMPILER=gcc 
     -DADIOS2_USE_CUDA=OFF 
     -DADIOS2_BUILD_EXAMPLES=OFF 
     -DBUILD_TESTING=OFF 
     -DADIOS2_USE_SST=OFF 
     -DKokkos_ROOT=${KOKKOS_HOME}/install 
     -D CMAKE_CUDA_ARCHITECTURES=70 
     -DCMAKE_CXX_COMPILER=${KOKKOS_HOME}/kokkos/bin/nvcc_wrapper
)
cmake "${ARGS[@]}" ..
make -j4
make install
