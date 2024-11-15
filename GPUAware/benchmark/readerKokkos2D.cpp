/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * sstReaderKokkos.cpp  Simple example of reading dataFloats through ADIOS2 SST
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
int read(adios2::ADIOS &adios, const std::string fname, const size_t Nx, const size_t Ny, const size_t Nz,
         const std::string engine)
{
    adios2::IO io = adios.DeclareIO("ReadIO");
    io.SetEngine(engine);
    if (engine == "DataMan")
        io.SetParameters({{"IPAddress", "127.0.0.1"}, {"Port", "12306"}, {"Timeout", "5"}});

    adios2::Engine engineReader = io.Open(fname, adios2::Mode::Read);
	size_t Lx = static_cast<size_t>(2560 / Nx);
	size_t Ly = static_cast<size_t>(960 / Ny);
	size_t Lz = static_cast<size_t>(3456 / Nz);
	const size_t index_z = rank % Nz;
	const size_t index_y = static_cast<size_t>(rank / Nz) % Ny;
	const size_t index_x = static_cast<size_t>(rank / (Nz * Ny));

    unsigned int step = 0;
    Kokkos::View<float ***[19], MemSpace> gpuSimData("simBuffer", Lx, Ly, Lz);
    ExecSpace exe_space;
    for (; engineReader.BeginStep() == adios2::StepStatus::OK; ++step)
    {
        auto data = io.InquireVariable<float>("dataFloats");
		const adios2::Dims start{Lx * index_x, Ly * index_y, Lz*index_z, 0};
		const adios2::Dims count{Lx, Ly, Lz, 19};
        const adios2::Box<adios2::Dims> sel(start, count);
        data.SetSelection(sel);

        auto tm_start = std::chrono::steady_clock::now();
        // var.SetMemorySpace(adios2::MemorySpace::GPU);
        engineReader.Get(data, gpuSimData.data());
        engineReader.EndStep();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-tm_start;
	    double get_time = elapsed_seconds.count();
	    double global_get_time = 0;
	    MPI_Reduce(&get_time, &global_get_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
        if (rank == 0)
        {
            std::cout << "Read4D " << engine << " " << exe_space.name() << " "
                      << 19 * Lx * Ly * Lz * sizeof(float) / (1024.*1024*1024) << " " << global_get_time
                      << " units:GB:s" << std::endl;
        }
    }
    engineReader.Close();
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0] << " engine inputFile size_decomp1 size_decomp2 size_decomp3 device/host" << std::endl;
        return 1;
    }
    int provided;
    // MPI_THREAD_MULTIPLE is only required if you enable the SST MPI_DP
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::string engine(argv[1]);
    if (rank == 0)
        std::cout << "Using engine " << engine << std::endl;

    const std::string filename = argv[2] ? argv[2] : engine + "WriteReadKokkos.bp";
    const unsigned int Nx = std::stoi(argv[3]);
    const unsigned int Ny = std::stoi(argv[4]);
    const unsigned int Nz = std::stoi(argv[5]);
    const std::string memorySpace = argv[6];

    Kokkos::initialize(argc, argv);
    {
        adios2::ADIOS adios(MPI_COMM_WORLD);

        if (memorySpace == "device" || memorySpace == "Device")
        {
            using mem_space = Kokkos::DefaultExecutionSpace::memory_space;
            if (rank == 0)
                std::cout << "Memory space: DefaultMemorySpace" << std::endl;
            read<mem_space, Kokkos::DefaultExecutionSpace>(adios, filename, Nx, Ny, Nz,
                                                             engine);
        }
        else
        {
            if (rank == 0)
                std::cout << "Memory space: HostSpace" << std::endl;
            read<Kokkos::HostSpace, Kokkos::Serial>(adios, filename, Nx, Ny, Nz, engine);
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
