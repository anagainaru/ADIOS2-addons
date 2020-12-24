#------------------------------------------------------------------------------#
# Distributed under the OSI-approved Apache License, Version 2.0.  See
# accompanying file Copyright.txt for details.
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.12)
project(AdiosCuda CXX)

find_package(Kokkos REQUIRED PATHS /home/gainaru/kokkos/install/lib/cmake)
find_package(adios2 REQUIRED PATHS /home/gainaru/adios/install/lib/cmake)

add_executable(adios.kokkos bpWriteRead.cpp)
target_link_libraries(adios.kokkos adios2::adios2 Kokkos::kokkos)
