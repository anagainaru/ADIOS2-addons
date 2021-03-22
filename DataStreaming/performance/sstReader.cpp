/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * helloSstReader.cpp
 *
 *  Created on: Aug 17, 2017
v *      Author: Greg Eisenhauer
 */

#include <chrono>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

#include <adios2.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " number_variables"
                  << std::endl;
        return -1;
    }
    const size_t variablesSize = atoi(argv[1]);

    int rank;
    int size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<float> myFloats;

    try
    {
        adios2::ADIOS adios(MPI_COMM_WORLD);

        adios2::IO sstIO = adios.DeclareIO("myIO");
        sstIO.SetEngine("Sst");

        adios2::Engine sstReader = sstIO.Open("helloSst", adios2::Mode::Read);

        double get_time = 0;
        int Nx = 0;
        auto start_step = std::chrono::steady_clock::now();
        sstReader.BeginStep();
        for (unsigned int v = 0; v < variablesSize; ++v)
        {
            std::string namev("sstFloats");
            namev += std::to_string(v);
            adios2::Variable<float> sstFloats =
                sstIO.InquireVariable<float>(namev);

            const std::size_t total_size = sstFloats.Shape()[0];
            const std::size_t my_start = (total_size / size) * rank;
            const std::size_t my_count = (total_size / size);
            const adios2::Dims pos_start{my_start};
            const adios2::Dims count{my_count};

            const adios2::Box<adios2::Dims> sel(pos_start, count);

            myFloats.resize(my_count);
            Nx = my_count;

            sstFloats.SetSelection(sel);
            auto start_get = std::chrono::steady_clock::now();
            sstReader.Get(sstFloats, myFloats.data());
            auto end_get = std::chrono::steady_clock::now();
            get_time += (end_get - start_get).count() / 1000;
        }
        sstReader.EndStep();
        auto end_step = std::chrono::steady_clock::now();
        std::cout << "SST,Read," << rank << ","  << Nx << ","
                  << variablesSize << "," << get_time << ","
                  << (end_step - start_step).count() / 1000 << std::endl;

        sstReader.Close();
    }
    catch (std::invalid_argument &e)
    {
        std::cout << "Invalid argument exception, STOPPING PROGRAM from rank "
                  << rank << "\n";
        std::cout << e.what() << "\n";
    }
    catch (std::ios_base::failure &e)
    {
        std::cout << "IO System base failure exception, STOPPING PROGRAM "
                     "from rank "
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
