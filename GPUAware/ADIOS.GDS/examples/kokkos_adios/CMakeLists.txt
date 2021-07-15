#------------------------------------------------------------------------------#
# Distributed under the OSI-approved Apache License, Version 2.0.  See
# accompanying file Copyright.txt for details.
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.12)
project(AdiosCuda CXX)

find_package(Kokkos REQUIRED)
find_package(adios2 REQUIRED)

add_executable(adios.kokkos bpWriteRead.cpp)
target_link_libraries(adios.kokkos adios2::adios2 Kokkos::kokkos)
