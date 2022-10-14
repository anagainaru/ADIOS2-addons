cmake_minimum_required(VERSION 3.12)
project(SSTadios C CXX)

find_package(adios2 REQUIRED)
find_package(MPI REQUIRED)

add_executable(sscReadWriter sscReadWrite.cpp)
target_link_libraries(sscReadWriter adios2::adios2 adios2::cxx11_mpi MPI::MPI_CXX)

add_executable(sstReader sstReader.cpp)
target_link_libraries(sstReader adios2::adios2 adios2::cxx11_mpi MPI::MPI_CXX)

add_executable(sstWriter sstWriter.cpp)
target_link_libraries(sstWriter adios2::adios2 adios2::cxx11_mpi MPI::MPI_CXX)

add_executable(dsReader dataSpacesReader.cpp)
target_link_libraries(dsReader adios2::adios2 adios2::cxx11_mpi MPI::MPI_CXX)

add_executable(dsWriter dataSpacesWriter.cpp)
target_link_libraries(dsWriter adios2::adios2 adios2::cxx11_mpi MPI::MPI_CXX)

add_executable(iReadWriter inlineReadWrite.cpp)
target_link_libraries(iReadWriter adios2::adios2 adios2::cxx11_mpi MPI::MPI_CXX)

add_executable(dmReader dataManReader.cpp)
target_link_libraries(dmReader adios2::adios2 adios2::cxx11_mpi MPI::MPI_CXX)

add_executable(dmWriter dataManWriter.cpp)
target_link_libraries(dmWriter adios2::adios2 adios2::cxx11_mpi MPI::MPI_CXX)
