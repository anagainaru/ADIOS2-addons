/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * helloSstWriter.cpp
 *
 *  Created on: Aug 17, 2017
 *      Author: Greg Eisenhauer
 */

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
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " array_size"
                  << std::endl;
        return -1;
    }
    const size_t Nx = atoi(argv[1]);

    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto myFloats = create_random_data(Nx);

    try
    {
        adios2::ADIOS adios(MPI_COMM_WORLD);
        adios2::IO sstIO = adios.DeclareIO("myIO");
        sstIO.SetEngine("Sst");

        // Define variable and local size
        auto bpFloats = sstIO.DefineVariable<float>("bpFloats", {size * Nx},
                                                    {rank * Nx}, {Nx});

        // Create engine smart pointer to Sst Engine due to polymorphism,
        // Open returns a smart pointer to Engine containing the Derived class
        adios2::Engine sstWriter = sstIO.Open("helloSst", adios2::Mode::Write);
        auto start_step = std::chrono::steady_clock::now();
        sstWriter.BeginStep();
        auto start_time = std::chrono::system_clock::now();
        auto start_put = std::chrono::steady_clock::now();
        sstWriter.Put<float>(bpFloats, myFloats.data());
        sstWriter.EndStep();
        auto end_step = std::chrono::steady_clock::now();
        // Time in miliseconds
        std::cout << "SST,Write," << rank << ","  << Nx << ","
                  << (end_step - start_step).count() / 1000 << ","
                  << (end_step - start_put).count() / 1000 << ","
                  << start_time.time_since_epoch().count() << std::endl;
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
