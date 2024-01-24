#------------------------------------------------------------------------------#
# Distributed under the OSI-approved Apache License, Version 2.0.  See
# accompanying file Copyright.txt for details.
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.12)
project(ADIOS2-Fortran-Kokkos-WriteRead)

# CXX Compiler settings only in for this example
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Kokkos 3.7)
if(DEFINED Kokkos_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "${Kokkos_CXX_COMPILER}")
endif()

include(CheckLanguage)
check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
  enable_language(Fortran)
endif()

find_package(MPI) 
if(MPI_FOUND)
  # Workaround for various MPI implementations forcing the link of C++ bindings
  add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)
endif()

find_package(adios2 REQUIRED)

find_package(flcl)

add_executable(adios2_writeread_f adiosWriteRead.F90 view-f.f90 view-cxx.cc)
target_link_libraries(adios2_writeread_f adios2::fortran_mpi MPI::MPI_Fortran flcl::flcl)
if (CMAKE_Fortran_COMPILER_ID STREQUAL "XL")
    target_link_options(adios2_writeread_f PRIVATE LINKER:-lxlf90_r)
endif()
if (CMAKE_Fortran_COMPILER_ID STREQUAL "Intel" OR CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set_target_properties(adios2_writeread_f PROPERTIES LINKER_LANGUAGE Fortran)
endif()
