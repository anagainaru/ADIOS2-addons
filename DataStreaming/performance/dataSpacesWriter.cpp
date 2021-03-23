#include <iostream>
#include <vector>
#include <random>
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
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto myFloats = create_random_data(Nx);

    try
    {
        adios2::ADIOS adios(MPI_COMM_WORLD);
        adios2::IO dataSpacesIO = adios.DeclareIO("myIO");
        dataSpacesIO.SetEngine("DATASPACES");

        // Define variable and local size
        std::vector<adios2::Variable<float>> dsFloats(variablesSize);
        for (unsigned int v = 0; v < variablesSize; ++v)
        {
            std::string namev("dsFloats");
            namev += std::to_string(v);
            dsFloats[v] = dataSpacesIO.DefineVariable<float>(
                    namev, {size * Nx}, {rank * Nx}, {Nx});
        }

        // Create engine smart pointer to Sst Engine due to polymorphism,
        // Open returns a smart pointer to Engine containing the Derived class
        adios2::Engine dataSpacesWriter =
            dataSpacesIO.Open("helloDataSpaces", adios2::Mode::Write);

        double put_time = 0;
        for (unsigned int timeStep = 0; timeStep < 1; ++timeStep)
        {
            auto start_step = std::chrono::steady_clock::now();
            dataSpacesWriter.BeginStep();
            for (unsigned int v = 0; v < variablesSize; ++v)
            {
                myFloats[rank] += static_cast<float>(v + rank);
                auto start_put = std::chrono::steady_clock::now();
                dataSpacesWriter.Put<float>(dsFloats[v], myFloats.data());
                auto end_put = std::chrono::steady_clock::now();
                put_time += (end_put - start_put).count() / 1000;
            }
            dataSpacesWriter.EndStep();
            auto end_step = std::chrono::steady_clock::now();
            // Time in microseconds
            std::cout << "DataSpaces,Write," << rank << ","  << Nx << ","
                      << variablesSize << "," << put_time << ","
                      << (end_step - start_step).count() / 1000 << std::endl;
        }
        dataSpacesWriter.Close();
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
