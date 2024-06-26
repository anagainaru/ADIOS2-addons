#include <chrono>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

#include <Kokkos_Core.hpp>

#include <adios2.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " array_size number_variables [process name]"
                  << std::endl;
        std::cout << "If the name is provided the code is ran in debug mode" << std::endl;
        return -1;
    }
    const size_t Nx = atoi(argv[1]);
    const size_t variablesSize = atoi(argv[2]);
    std::string name;
    unsigned int debug = 0;
    if (argc > 3)
    {
        name = argv[3];
        debug = 1;
    }

    int rank = 0, size = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Kokkos::initialize( argc, argv );
    {
        adios2::ADIOS adios("adios2.xml", MPI_COMM_WORLD);
        adios2::IO sstIO = adios.DeclareIO("sstOnDemand");

        adios2::Engine sstReader = sstIO.Open("helloSst", adios2::Mode::Read);

        double get_time = 0;
        const std::size_t my_start = Nx * rank;
        const adios2::Dims pos_start{my_start};
        const adios2::Dims count{Nx};
        const adios2::Box<adios2::Dims> sel(pos_start, count);

        auto start_step = std::chrono::steady_clock::now();
        int steps = 0;
        using mem_space = Kokkos::DefaultExecutionSpace::memory_space;
        Kokkos::View<float *, mem_space> myFloats("simBuffer", variablesSize * Nx);
        while (sstReader.BeginStep() == adios2::StepStatus::OK)
        {
            size_t currentStep = sstReader.CurrentStep();
            for (unsigned int v = 0; v < variablesSize; ++v)
            {
                std::string namev("sstFloats");
                namev += std::to_string(v);
                adios2::Variable<float> sstFloats =
                    sstIO.InquireVariable<float>(namev);

                sstFloats.SetSelection(sel);
                auto start_get = std::chrono::steady_clock::now();
                sstReader.Get(sstFloats, myFloats.data() + (v * Nx));
                auto end_get = std::chrono::steady_clock::now();
                get_time += (end_get - start_get).count() / 1000;
            }
            sstReader.EndStep();
            auto simData = Kokkos::create_mirror_view_and_copy(
              Kokkos::HostSpace{}, myFloats);
            steps += 1;
            if (debug == 1){
                for (unsigned int v = 0; v < variablesSize; ++v)
                {
                    std::cout << name << ": Get step " << currentStep
                        << " variable" << v << " " << simData[v * Nx]
                        << std::endl;
                }
            }
        }
        auto end_step = std::chrono::steady_clock::now();
        double total_time = (end_step - start_step).count() / (size * 1000);
        get_time /= size;

        double global_get_sum = 0;
        MPI_Reduce(&get_time, &global_get_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
        double global_sum = 0;
        MPI_Reduce(&total_time, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);

        // Time in microseconds
        if (rank == 0){
            std::cout << "SST,Read," << size << "," << Nx << ","
                  << variablesSize << "," << steps << "," << global_get_sum
                  << "," << global_sum  << std::endl;
	    }
        sstReader.Close();
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
