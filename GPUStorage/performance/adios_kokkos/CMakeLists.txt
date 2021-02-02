#------------------------------------------------------------------------------#
# Distributed under the OSI-approved Apache License, Version 2.0.  See
# accompanying file Copyright.txt for details.
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.12)
project(AdiosKokkos CXX)

find_package(Kokkos REQUIRED)
find_package(adios2 REQUIRED)

add_executable(adios.kokkos bpWriteRead.cpp)
target_include_directories(adios.kokkos PUBLIC /usr/local/cuda-11.1/targets/x86_64-linux/lib/)
target_link_directories(adios.kokkos PUBLIC /usr/local/cuda-11.1/targets/x86_64-linux/lib/)
target_link_libraries(adios.kokkos adios2::adios2 Kokkos::kokkos -lcufile)
