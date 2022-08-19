#------------------------------------------------------------------------------#
# Distributed under the OSI-approved Apache License, Version 2.0.  See
# accompanying file Copyright.txt for details.
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.18)
find_package(adios2)
find_package(Kokkos)

add_executable(KokkosBP4WriteRead kokkosBP4.cpp)
target_link_libraries(KokkosBP4WriteRead PUBLIC adios2::cxx11 Kokkos::kokkos)
