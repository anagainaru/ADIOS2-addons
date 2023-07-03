#include <iostream>
#include <vector>
#include <random>    
#include <algorithm> 
#include <functional>
#include <chrono>

#include<unistd.h>

#include <Kokkos_Random.hpp>
#include <Kokkos_Core.hpp>

#include <adios2.h>
#include <mpi.h>



Kokkos::View<float *> create_random_data(int n)
{
    using RandomPool =
        Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;
    RandomPool random_pool(5374857);
    Kokkos::View<float *> v("simData", n);

    Kokkos::parallel_for(
        "create_random_data", Kokkos::RangePolicy<>(0, n),
        KOKKOS_LAMBDA(int i) {
            RandomPool::generator_type generator = random_pool.get_state();
            v(i) = generator.frand(0.f, 1.f);
            random_pool.free_state(generator);
        });
    return v;
}

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
    int total_steps = 10;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Kokkos::initialize( argc, argv );
    {
        auto myFloats = create_random_data(Nx * variablesSize);

        adios2::ADIOS adios("adios2.xml", MPI_COMM_WORLD);
        adios2::IO sstIO = adios.DeclareIO("sstOnDemand");

        std::vector<adios2::Variable<float>> sstFloats(variablesSize);
        for (unsigned int v = 0; v < variablesSize; ++v)
        {
            std::string namev("sstFloats");
            namev += std::to_string(v);
            sstFloats[v] = sstIO.DefineVariable<float>(namev, {size * Nx},
                                                      {rank * Nx}, {Nx});
        }

        adios2::Engine sstWriter = sstIO.Open("helloSst", adios2::Mode::Write);
        double put_time = 0;
        auto start_step = std::chrono::steady_clock::now();
        for (unsigned int timeStep = 0; timeStep < total_steps; ++timeStep)
        {
            sstWriter.BeginStep();
            for (unsigned int v = 0; v < variablesSize; ++v)
            {
                myFloats[v * Nx] = v + timeStep * variablesSize;
                auto start_put = std::chrono::steady_clock::now();
                sstWriter.Put<float>(sstFloats[v], myFloats.data() + v * Nx);
                auto end_put = std::chrono::steady_clock::now();
                sleep(1);
                put_time += (end_put - start_put).count() / 1000;
                if (debug == 1){
                    std::cout << "p0: Put step " << timeStep << " variable"
                        << v << " " << myFloats[v * Nx] << std::endl;
                }
            }
            sstWriter.EndStep();
        }
        auto end_step = std::chrono::steady_clock::now();
        double total_time = (end_step - start_step).count() / 1000;

	    double global_put_sum;
	    MPI_Reduce(&put_time, &global_put_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
		   MPI_COMM_WORLD);
	    double global_sum;
	    MPI_Reduce(&total_time, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
		   MPI_COMM_WORLD);

	    // Time in microseconds
	    if (rank == 0)
		std::cout << "SST,Write," << size << "," << Nx << ","
			  << variablesSize << "," << total_steps << ","
              <<  global_put_sum / size << "," << global_sum / size
              << std::endl;
        sstWriter.Close();
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
