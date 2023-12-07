cmake_minimum_required(VERSION 3.12)
project(ADIOS2PerformanceGPU C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

## Kokkos ##
find_package(Kokkos REQUIRED)
if(DEFINED Kokkos_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${Kokkos_CXX_COMPILER}")
	message("Compiler used: ${Kokkos_CXX_COMPILER}")
endif()

if (BUILD_BENCHMARKS)
## ADIOS2 ##
find_package(adios2 REQUIRED)

### MPI ###
find_package(MPI REQUIRED)

## Dataman Code ##
find_package(ZeroMQ 4.1 QUIET)

add_executable(adios2_writerKokkos1D writerKokkos1D.cpp)
add_executable(adios2_readerKokkos1D readerKokkos1D.cpp)
kokkos_compilation(SOURCE writerKokkos1D.cpp)
kokkos_compilation(SOURCE readerKokkos1D.cpp)
target_link_libraries(adios2_writerKokkos1D adios2::adios2 adios2::cxx11_mpi MPI::MPI_CXX Kokkos::kokkos)
target_link_libraries(adios2_readerKokkos1D adios2::adios2 adios2::cxx11_mpi MPI::MPI_CXX Kokkos::kokkos)

add_executable(adios2_writerKokkos2D writerKokkos2D.cpp)
add_executable(adios2_readerKokkos2D readerKokkos2D.cpp)
kokkos_compilation(SOURCE writerKokkos2D.cpp)
kokkos_compilation(SOURCE readerKokkos2D.cpp)
target_link_libraries(adios2_writerKokkos2D adios2::adios2 adios2::cxx11_mpi MPI::MPI_CXX Kokkos::kokkos)
target_link_libraries(adios2_readerKokkos2D adios2::adios2 adios2::cxx11_mpi MPI::MPI_CXX Kokkos::kokkos)
endif()

if (BUILD_KERNELS)
    add_executable(perf_gpukernel GPUKernelPerf.cpp)
    kokkos_compilation(SOURCE GPUKernelPerf.cpp)
    target_link_libraries(perf_gpukernel Kokkos::kokkos)
endif()
