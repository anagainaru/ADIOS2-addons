/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * Simple example of writing and reading dataFloats through an ADIOS2
 * engine with multiple simulations steps for every IO step using Kokkos
 */
#include <ios>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <chrono>

#include <mpi.h>
#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>
#include <Kokkos_Core.hpp>

int rank, size;

template <class MemSpace, class ExecSpace>
void writer(adios2::ADIOS &adios, const std::string &engine, const std::string &fname,
            const size_t Nx, unsigned int nSteps, const bool include_copy_to_device)
{
	int internal_rank = rank;
	int internal_size = size;
    Kokkos::View<float *, MemSpace> gpuSimData("simBuffer", Nx);
    static_assert(Kokkos::SpaceAccessibility<ExecSpace, MemSpace>::accessible, "");
    Kokkos::parallel_for(
        "initBuffer", Kokkos::RangePolicy<ExecSpace>(0, Nx),
        KOKKOS_LAMBDA(int i) { gpuSimData(i) = static_cast<float>(i * internal_rank); });
    Kokkos::fence();

    // Set up the ADIOS structures
    adios2::IO adIO = adios.DeclareIO("WriteIO");
    adIO.SetEngine(engine);
    if (engine == "DataMan")
    {
        adIO.SetParameters({{"IPAddress", "127.0.0.1"},
                             {"Port", "12306"},
                             {"Timeout", "5"},
                             {"RendezvousReaderCount", "1"}});
    }

    const adios2::Dims shape{static_cast<size_t>(size * Nx)};
    const adios2::Dims start{static_cast<size_t>(rank * Nx)};
    const adios2::Dims count{Nx};
    auto dataFloats = adIO.DefineVariable<float>("dataFloats", shape, start, count);

    adios2::Engine engineWriter = adIO.Open(fname, adios2::Mode::Write);

    ExecSpace exe_space;
    for (unsigned int step = 0; step < nSteps; ++step)
    {
        auto tm_start = std::chrono::steady_clock::now();
        engineWriter.BeginStep();
        // var.SetMemorySpace(adios2::MemorySpace::GPU);
        engineWriter.Put(dataFloats, gpuSimData.data());
        engineWriter.EndStep();
        auto tm_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = tm_end - tm_start;

	    double put_time = elapsed_seconds.count();
	    double global_put_time = 0;
	    MPI_Reduce(&put_time, &global_put_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
        std::cout << "Write1D " << engine << " " << exe_space.name() << " "
                  << Nx * sizeof(float) / (1024*1024) << " " << global_put_time
                  << " units:MB:s " << std::endl;
        // Update values in the simulation dataFloats
        Kokkos::parallel_for(
            "updateBuffer", Kokkos::RangePolicy<ExecSpace>(0, Nx),
            KOKKOS_LAMBDA(int i) { gpuSimData(i) += (10 / internal_size); });
        Kokkos::fence();
    }
    engineWriter.Close();
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0] << " engine array_size steps device/host [outputFile]" << std::endl;
        return 1;
    }
    int provided;
    // MPI_THREAD_MULTIPLE is only required if you enable the SST MPI_DP
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Kokkos::initialize(argc, argv);

    const std::string engine(argv[1]);
    std::cout << "Engine: " << engine << std::endl;

    const std::string filename = argv[5] ? argv[5] : engine + "StepsWriteReadCuda";
    const unsigned int nSteps = std::stoi(argv[3]);
    const unsigned int Nx = std::stoi(argv[2]);
    const std::string memorySpace = argv[4];
    try
    {
        /** ADIOS class factory of IO class objects */
        adios2::ADIOS adios(MPI_COMM_WORLD);
        if (memorySpace == "Device")
        {
            std::cout << "Memory space: DefaultMemorySpace" << std::endl;
            using mem_space = Kokkos::DefaultExecutionSpace::memory_space;
            writer<mem_space, Kokkos::DefaultExecutionSpace>(adios, engine, filename, Nx, nSteps, false);
        }
        if (memorySpace == "Host")
        {
            std::cout << "Memory space: HostSpace" << std::endl;
            writer<Kokkos::HostSpace, Kokkos::Serial>(adios, engine, filename, Nx, nSteps, true);
        }
    }
    catch (std::invalid_argument &e)
    {
        std::cout << "Invalid argument exception, STOPPING PROGRAM\n";
        std::cout << e.what() << "\n";
    }
    catch (std::ios_base::failure &e)
    {
        std::cout << "IO System base failure exception, STOPPING PROGRAM\n";
        std::cout << e.what() << "\n";
    }
    catch (std::exception &e)
    {
        std::cout << "Exception, STOPPING PROGRAM\n";
        std::cout << e.what() << "\n";
    }
    Kokkos::finalize();

    MPI_Finalize();
    return 0;
}
