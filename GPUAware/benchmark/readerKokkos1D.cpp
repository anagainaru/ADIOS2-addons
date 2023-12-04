/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * bpStepsWriteReadKokkos.cpp  Simple example of writing and reading dataFloats through ADIOS2 BP
 * engine with multiple simulations steps for every IO step using Kokkos
 */
#include <ios>
#include <iostream>
#include <stdexcept> //std::invalid_argument std::exception
#include <vector>
#include <chrono>

#include <mpi.h>
#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>
#include <Kokkos_Core.hpp>

int rank, size;

template <class MemSpace, class ExecSpace>
void reader(adios2::ADIOS &adios, const std::string &engine, const std::string &fname,
            const size_t Nx, const bool include_copy_to_device)
{
    adios2::IO adIO = adios.DeclareIO("ReadIO");
    adIO.SetEngine(engine);
    if (engine == "DataMan")
        adIO.SetParameters({{"IPAddress", "127.0.0.1"}, {"Port", "12306"}, {"Timeout", "5"}});

    adios2::Engine engineReader = adIO.Open(fname, adios2::Mode::Read);

    Kokkos::View<float *, MemSpace> gpuSimData("simBuffer", Nx);
    ExecSpace exe_space;
    for (unsigned int step = 0; engineReader.BeginStep() == adios2::StepStatus::OK; ++step)
    {
        auto tm_start = std::chrono::steady_clock::now();
        auto dataFloats = adIO.InquireVariable<float>("dataFloats");
        if (! dataFloats) continue;
        const adios2::Dims start{Nx * rank};
        const adios2::Dims count{Nx};
        const adios2::Box<adios2::Dims> sel(start, count);
        dataFloats.SetSelection(sel);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-tm_start;
	    double get_time = elapsed_seconds.count();
	    double global_get_time = 0;
	    MPI_Reduce(&get_time, &global_get_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
        if (rank == 0)
        {
            std::cout << "Inquire " << engine << " " << exe_space.name() << " "
                      << Nx * sizeof(float) / (1024.*1024) << " " << global_get_time
                      << " units:MB:s " << std::endl;
        }

        tm_start = std::chrono::steady_clock::now();
        // var.SetMemorySpace(adios2::MemorySpace::GPU);
        engineReader.Get(dataFloats, gpuSimData.data());
        engineReader.EndStep();
        end = std::chrono::steady_clock::now();
        elapsed_seconds = end-tm_start;

	    get_time = elapsed_seconds.count();
	    global_get_time = 0;
	    MPI_Reduce(&get_time, &global_get_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
        if (rank == 0)
        {
            std::cout << "Read " << engine << " " << exe_space.name() << " "
                      << Nx * sizeof(float) / (1024.*1024) << " " << global_get_time
                      << " " << Nx * sizeof(float) / (1024. * 1024 * 1024 * global_get_time)
                      << " units:MB:s:GB/s" << std::endl;
        }
        // measure the time to copy the data from the host to the GPU
        if (include_copy_to_device)
        {
            tm_start = std::chrono::steady_clock::now();
            auto simData = Kokkos::create_mirror_view_and_copy(
              Kokkos::DefaultExecutionSpace(), gpuSimData);
            end = std::chrono::steady_clock::now();
            get_time = (end - tm_start).count();
            global_get_time = 0;
            MPI_Reduce(&get_time, &global_get_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                       MPI_COMM_WORLD);
            if (rank == 0)
            {
                std::cout << "DeepCpy " << engine << " " << exe_space.name() << " "
                          << Nx * sizeof(float) / (1024.*1024) << " " << global_get_time
                      << " units:MB:s " << std::endl;
            }
         }
    }

    engineReader.Close();
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0] << " engine inputFile array_size device/host" << std::endl;
        return 1;
    }
    int provided;
    // MPI_THREAD_MULTIPLE is only required if you enable the SST MPI_DP
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Kokkos::initialize(argc, argv);

    const std::string engine(argv[1]);
    if (rank == 0)
        std::cout << "Using engine " << engine << std::endl;

    const std::string filename = argv[2] ? argv[2] : engine + "WriteReadKokkos.bp";
    const unsigned int Nx = std::stoi(argv[3]);
    const std::string memorySpace = argv[4];
    try
    {
        /** ADIOS class factory of IO class objects */
        adios2::ADIOS adios(MPI_COMM_WORLD);
        if (memorySpace == "device" || memorySpace == "Device")
        {
            if (rank == 0)
                std::cout << "Memory space: DefaultMemorySpace" << std::endl;
            using mem_space = Kokkos::DefaultExecutionSpace::memory_space;
            reader<mem_space, Kokkos::DefaultExecutionSpace>(adios, engine, filename, Nx, false);
        }
        if (memorySpace == "host" || memorySpace == "Host")
        {
            if (rank == 0)
                std::cout << "Memory space: HostSpace" << std::endl;
            reader<Kokkos::HostSpace, Kokkos::Serial>(adios, engine, filename, Nx, true);
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
