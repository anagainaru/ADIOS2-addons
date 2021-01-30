#------------------------------------------------------------------------------#
# Distributed under the OSI-approved Apache License, Version 2.0.  See
# accompanying file Copyright.txt for details.
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.12)
project(AdiosCuda CXX)

find_package(CUDA REQUIRED)
find_package(adios2 REQUIRED)

add_executable(adios.cuda bpWriteRead.cpp)
target_include_directories(adios.cuda PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(adios.cuda PUBLIC adios2::adios2 ${CUDA_LIBRARIES})
