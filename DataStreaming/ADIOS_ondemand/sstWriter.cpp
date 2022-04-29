#include <iostream>
#include <vector>
#include <random>    
#include <algorithm> 
#include <functional>
#include <chrono>

#include <adios2.h>
#include <mpi.h>

std::vector<float> create_random_data(int n) {
    std::random_device r;
    std::seed_seq      seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937       eng(seed);

    std::uniform_int_distribution<int> dist;
    std::vector<float> v(n);

    generate(begin(v), end(v), bind(dist, eng));
    return v;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " array_size number_variables"
                  << std::endl;
        return -1;
    }
    const size_t Nx = atoi(argv[1]);
    const size_t variablesSize = atoi(argv[2]);

    int rank = 0, size = 1;
    int total_steps = 100;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto myFloats = create_random_data(Nx);

    try
    {
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

        // Create engine smart pointer to Sst Engine due to polymorphism,
        // Open returns a smart pointer to Engine containing the Derived class
        adios2::Engine sstWriter = sstIO.Open("helloSst", adios2::Mode::Write);
        double put_time = 0;
        auto start_step = std::chrono::steady_clock::now();
        for (unsigned int timeStep = 0; timeStep < total_steps; ++timeStep)
        {
            sstWriter.BeginStep();
            for (unsigned int v = 0; v < variablesSize; ++v)
            {
                myFloats[rank] += static_cast<float>(v + rank);
                auto start_put = std::chrono::steady_clock::now();
                sstWriter.Put<float>(sstFloats[v], myFloats.data());
                auto end_put = std::chrono::steady_clock::now();
                put_time += (end_put - start_put).count() / 1000;
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
    catch (std::invalid_argument &e)
    {
        std::cout << "Invalid argument exception, STOPPING PROGRAM from rank "
                  << rank << "\n";
        std::cout << e.what() << "\n";
    }
    catch (std::ios_base::failure &e)
    {
        std::cout
            << "IO System base failure exception, STOPPING PROGRAM from rank "
            << rank << "\n";
        std::cout << e.what() << "\n";
    }
    catch (std::exception &e)
    {
        std::cout << "Exception, STOPPING PROGRAM from rank " << rank << "\n";
        std::cout << e.what() << "\n";
    }

    MPI_Finalize();
    return 0;
}
