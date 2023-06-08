cmake_minimum_required(VERSION 3.12)
project(AdiosCudaExample LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(adios2 REQUIRED)
enable_language(CUDA)

add_executable(adiosCUDAex cudaADIOS2example.cpp)
target_sources(adiosCUDAex PRIVATE cudaRoutines.cu)
target_link_libraries(adiosCUDAex PUBLIC adios2::adios2 adios2::cxx11 CUDA::cudart)
