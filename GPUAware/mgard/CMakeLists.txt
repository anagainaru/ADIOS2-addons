cmake_minimum_required(VERSION 3.18)

find_package(adios2)
find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)

add_executable(MGARDWriteRead example_adios2_mgard.cu)
target_link_libraries(MGARDWriteRead PUBLIC adios2::cxx11 CUDA::cudart)
set_target_properties(MGARDWriteRead PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
