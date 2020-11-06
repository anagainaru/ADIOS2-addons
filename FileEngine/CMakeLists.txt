#------------------------------------------------------------------------------#
# Distributed under the OSI-approved Apache License, Version 2.0.  See
# accompanying file Copyright.txt for details.
#------------------------------------------------------------------------------#

add_executable(bpAnaWriteRead bpWriteRead.cpp)
target_link_libraries(bpAnaWriteRead adios2::cxx11)

if(ADIOS2_HAVE_MPI)
    add_executable(bpAnaWriteRead_mpi bpWriteRead.cpp)
    target_link_libraries(bpAnaWriteRead_mpi adios2::cxx11_mpi MPI::MPI_C)
endif()
