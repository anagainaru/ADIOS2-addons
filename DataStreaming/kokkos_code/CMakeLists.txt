cmake_minimum_required(VERSION 3.12)
project(SSTadios C CXX)

find_package(adios2 REQUIRED)
find_package(MPI REQUIRED)
find_package(Kokkos REQUIRED)

add_executable(sstReader sstReader.cpp)
target_link_libraries(sstReader adios2::adios2 adios2::cxx11_mpi MPI::MPI_CXX Kokkos::kokkos)

add_executable(sstWriter sstWriter.cpp)
target_link_libraries(sstWriter adios2::adios2 adios2::cxx11_mpi MPI::MPI_CXX Kokkos::kokkos)
