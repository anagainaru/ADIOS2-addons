/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * SscReadWriter.cpp
 *
 *  Created on: Feb 19, 2021
 *      Author: Ana Gainaru
 */

#include <iostream>
#include <vector>

#include <adios2.h>
#include <mpi.h>

void read_data(adios2::IO sscIO, int rank, int size)
{
    adios2::Engine sscReader = sscIO.Open("helloSsc", adios2::Mode::Read);
    sscReader.LockReaderSelections();
    sscReader.BeginStep();
    adios2::Variable<float> bpFloats =
        sscIO.InquireVariable<float>("bpFloats");
    std::cout << "Incoming variable is of size " << bpFloats.Shape()[0]
              << "\n";
    const std::size_t total_size = bpFloats.Shape()[0];
    const std::size_t my_start = (total_size / size) * rank;
    const std::size_t my_count = (total_size / size);

    const adios2::Dims start{my_start};
    const adios2::Dims count{my_count};

    const adios2::Box<adios2::Dims> sel(start, count);

    std::vector<float> myFloats;
    myFloats.resize(my_count);
    bpFloats.SetSelection(sel);
    sscReader.Get(bpFloats, myFloats.data());
    sscReader.EndStep();

    std::cout << "Reader rank " << rank << " reading " << my_count
              << " floats starting at element " << my_start << ":"
              << " first element " << myFloats.data()[0] << "\n";
    sscReader.Close();
}

void write_data(adios2::IO sscIO, int rank, int size)
{
    std::vector<float> myFloats = {
        (float)10.0 * rank + 10, (float)10.0 * rank + 11, (float)10.0 * rank + 12,
        (float)10.0 * rank + 13, (float)10.0 * rank + 14, (float)10.0 * rank + 15,
        (float)10.0 * rank + 16, (float)10.0 * rank + 17, (float)10.0 * rank + 18,
        (float)10.0 * rank + 19};
    const std::size_t Nx = myFloats.size();

    // Define variable and local size
    auto bpFloats = sscIO.DefineVariable<float>("bpFloats", {size * Nx},
                                                {rank * Nx}, {Nx});

    adios2::Engine sscWriter = sscIO.Open("helloSsc", adios2::Mode::Write);
    sscWriter.LockWriterDefinitions();
    sscWriter.BeginStep();
    sscWriter.Put<float>(bpFloats, myFloats.data());
    sscWriter.EndStep();
    sscWriter.Close();
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm mpiComm;
    int worldRank, worldSize;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    int mpiGroup = worldRank / (worldSize / 2);
    MPI_Comm_split(MPI_COMM_WORLD, mpiGroup, worldRank, &mpiComm);
    MPI_Comm_rank(mpiComm, &rank);
    MPI_Comm_size(mpiComm, &size);

    try
    {
        adios2::ADIOS adios(mpiComm);
        adios2::IO sscIO = adios.DeclareIO("myIO");
        sscIO.SetEngine("Ssc");

        if (mpiGroup==0)
            write_data(sscIO, rank, size);
        if (mpiGroup == 1)
            read_data(sscIO, rank, size);
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
