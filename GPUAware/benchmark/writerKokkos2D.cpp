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
int writer(adios2::ADIOS &adios, const std::string fname, const size_t Nx, const size_t Ny, const size_t Nz,
            const size_t nSteps, const std::string engine)
{
    // Initialize the simulation data
	int internal_rank = rank;
	int internal_size = size;
	size_t Lx = static_cast<size_t>(2560 / Nx);
	size_t Ly = static_cast<size_t>(960 / Ny);
	size_t Lz = static_cast<size_t>(3456 / Nz);
    Kokkos::View<float ***[19], MemSpace> gpuSimData("simBuffer", Lx, Ly, Lz);
    static_assert(Kokkos::SpaceAccessibility<ExecSpace, MemSpace>::accessible, "");
    Kokkos::parallel_for(
        "initBuffer",
		Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {Lx, Ly, Lz}),
		KOKKOS_LAMBDA(int x, int y, int z) {
            for (int i = 0; i < 19; i++)
                gpuSimData(x, y, z, i) = static_cast<float>(i * internal_rank + x + y + z);
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
	io.SetParameters({{"AggregationType", "EveryoneWrites"},
					  {"NumAggregators", "900"},
					  {"StatsLevel", "1"}});

	const size_t index_z = rank % Nz;
	const size_t index_y = static_cast<size_t>(rank / Nz) % Ny;
	const size_t index_x = static_cast<size_t>(rank / (Nz * Ny));
	std::cout << "[debug] Rank " << rank << " Index " << index_x << " " << index_y << " " << index_z << std::endl;

    const adios2::Dims shape{2560, 960, 3456, 19};
    const adios2::Dims start{Lx * index_x, Ly * index_y, Lz*index_z, 0};
    const adios2::Dims count{Lx, Ly, Lz, 19};
    auto data = io.DefineVariable<float>("dataFloats", shape, start, count);

    adios2::Engine engineWriter = io.Open(fname, adios2::Mode::Write);

    ExecSpace exe_space;
    for (int step = 0; step < nSteps; ++step)
    {
        auto tm_start = std::chrono::steady_clock::now();
        engineWriter.BeginStep();
        engineWriter.Put(data, gpuSimData.data());
        engineWriter.EndStep();
        auto tm_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = tm_end - tm_start;

	    double put_time = elapsed_seconds.count();
	    double global_put_time = 0;
	    MPI_Reduce(&put_time, &global_put_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
        if (rank == 0)
        {
            std::cout << "Write4D " << engine << " " << exe_space.name() << " "
                      << 19 * Lx * Ly * Lz * sizeof(float) / (1024.*1024*1024) << " " << global_put_time
                      << " units:GB:s" << std::endl;
        }

        // Update values in the simulation data
        Kokkos::parallel_for(
            "updateBuffer",
			Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {Lx, Ly, Lz}),
			KOKKOS_LAMBDA(int x, int y, int z) {
                for (int i = 0; i < 19; i++)
                    gpuSimData(x, y, z, i) += (10 / internal_size);
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
        std::cout << "Usage: " << argv[0] << " engine size_decomp1 size_decomp2 size_decomp3 steps device/host [outputFile]" << std::endl;
        return 1;
    }
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::string engine(argv[1]);
    if (rank == 0)
        std::cout << "Engine: " << engine << std::endl;

    const std::string filename = argv[7] ? argv[7] : engine + "StepsWriteReadCuda";
    const unsigned int nSteps = std::stoi(argv[5]);
    const unsigned int Nx = std::stoi(argv[2]);
    const unsigned int Ny = std::stoi(argv[3]);
    const unsigned int Nz = std::stoi(argv[4]);
    const std::string memorySpace = argv[6];

    Kokkos::initialize(argc, argv);
    {
        adios2::ADIOS adios(MPI_COMM_WORLD);
        if (memorySpace == "device" || memorySpace == "Device")
        {
            using mem_space = Kokkos::DefaultExecutionSpace::memory_space;
            if (rank == 0)
                std::cout << "Memory space: DefaultMemorySpace" << std::endl;
            writer<mem_space, Kokkos::DefaultExecutionSpace>(adios, filename, Nx, Ny, Nz,
                                                              nSteps, engine);
        }
        else
        {
            if (rank == 0)
                std::cout << "Memory space: HostSpace" << std::endl;
            writer<Kokkos::HostSpace, Kokkos::Serial>(adios, filename, Nx, Ny, Nz, nSteps,
                                                       engine);
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
