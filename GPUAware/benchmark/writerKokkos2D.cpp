/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * sstWriterKokkos.cpp  Simple example of writing dataFloats through ADIOS2 SST
 * engine with multiple simulations steps for every IO step using Kokkos
 */
#include <ios>
#include <iostream>
#include <vector>
#include <chrono>

#include <mpi.h>
#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>
#include <Kokkos_Core.hpp>

int rank, size;

template <class MemSpace, class ExecSpace>
int writer(adios2::ADIOS &adios, const std::string fname, const size_t Nx, const size_t Ny,
            const size_t nSteps, const std::string engine)
{
    // Initialize the simulation data
    Kokkos::View<float **, MemSpace> gpuSimData("simBuffer", Nx, Ny);
    static_assert(Kokkos::SpaceAccessibility<ExecSpace, MemSpace>::accessible, "");
    Kokkos::parallel_for(
        "initBuffer", Kokkos::RangePolicy<ExecSpace>(0, Nx), KOKKOS_LAMBDA(int i) {
            for (int j = 0; j < Ny; j++)
                gpuSimData(i, j) = static_cast<float>(j + i * rank);
        });
    Kokkos::fence();

    adios2::IO io = adios.DeclareIO("WriteIO");
    io.SetEngine(engine);
    if (engine == "DataMan")
    {
        io.SetParameters({{"IPAddress", "127.0.0.1"},
                             {"Port", "12306"},
                             {"Timeout", "5"},
                             {"RendezvousReaderCount", "1"}});
    }

    const adios2::Dims shape{Nx, size * Ny};
    const adios2::Dims start{0, rank * Ny};
    const adios2::Dims count{Nx, Ny};
    auto data = io.DefineVariable<float>("dataFloats", shape, start, count);

    adios2::Engine engineWriter = io.Open(fname, adios2::Mode::Write);

    ExecSpace exe_space;
    for (int step = 0; step < nSteps; ++step)
    {
        auto tm_start = std::chrono::steady_clock::now();
        engineWriter.BeginStep();
        // var.SetMemorySpace(adios2::MemorySpace::GPU);
        engineWriter.Put(data, gpuSimData.data());
        engineWriter.EndStep();
        auto tm_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = tm_end - tm_start;

	    double put_time = elapsed_seconds.count();
	    double global_put_time = 0;
	    MPI_Reduce(&put_time, &global_put_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
        std::cout << "Write2D " << engine << " " << exe_space.name() << " "
                  << Nx * sizeof(float) / (1024*1024) << " " << global_put_time
                  << " units:MB:s " << std::endl;

        // Update values in the simulation data
        Kokkos::parallel_for(
            "updateBuffer", Kokkos::RangePolicy<ExecSpace>(0, Nx), KOKKOS_LAMBDA(int i) {
                for (int j = 0; j < Ny; j++)
                    gpuSimData(i, j) += (10 / size);
            });
        Kokkos::fence();
    }

    engineWriter.Close();
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0] << " engine size_dim1 size_dim2 steps device/host [outputFile]" << std::endl;
        return 1;
    }
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::string engine(argv[1]);
    std::cout << "Engine: " << engine << std::endl;

    const std::string filename = argv[6] ? argv[6] : engine + "StepsWriteReadCuda";
    const unsigned int nSteps = std::stoi(argv[4]);
    const unsigned int Nx = std::stoi(argv[2]);
    const unsigned int Ny = std::stoi(argv[3]);
    const std::string memorySpace = argv[5];

    Kokkos::initialize(argc, argv);
    {
        adios2::ADIOS adios;

        std::cout << "Using engine " << engine << std::endl;
        if (memorySpace == "Device")
        {
            using mem_space = Kokkos::DefaultExecutionSpace::memory_space;
            std::cout << "Memory space: DefaultMemorySpace" << std::endl;
            writer<mem_space, Kokkos::DefaultExecutionSpace>(adios, filename, Nx, Ny,
                                                              nSteps, engine);
        }
        else
        {
            std::cout << "Memory space: HostSpace" << std::endl;
            writer<Kokkos::HostSpace, Kokkos::Serial>(adios, filename, Nx, Ny, nSteps,
                                                       engine);
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
