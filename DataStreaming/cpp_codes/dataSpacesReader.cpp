#include <chrono>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

#include <adios2.h>
#include <mpi.h>

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

    adios2::ADIOS adios(MPI_COMM_WORLD);
    const std::size_t my_start = Nx * rank;
    const adios2::Dims start{my_start};
    const adios2::Dims count{Nx};
    const adios2::Box<adios2::Dims> sel(start, count);

    std::vector<std::vector<float>> myFloats(variablesSize);
    for (unsigned int v = 0; v < variablesSize; v++)
    {
        myFloats[v].resize(Nx);
    }

    try
    {
        adios2::IO dataSpacesIO = adios.DeclareIO("myIO");
        dataSpacesIO.SetEngine("DATASPACES");

        adios2::Engine dataSpacesReader =
            dataSpacesIO.Open("helloDataSpaces", adios2::Mode::Read);

        double get_time = 0;
        auto start_step = std::chrono::steady_clock::now();
        dataSpacesReader.BeginStep();
        for (unsigned int v = 0; v < variablesSize; v++)
        {
            std::string namev("dsFloats");
            namev += std::to_string(v);
            adios2::Variable<float> dsFloats =
                dataSpacesIO.InquireVariable<float>(namev);

            dsFloats.SetSelection(sel);
            auto start_get = std::chrono::steady_clock::now();
            dataSpacesReader.Get(dsFloats, myFloats[v].data());
            auto end_get = std::chrono::steady_clock::now();
            get_time += (end_get - start_get).count() / 1000;
        }
        dataSpacesReader.EndStep();
        auto end_step = std::chrono::steady_clock::now();
        // Time in microseconds
        std::cout << "DataSpaces,Read," << rank << ","  << Nx << ","
                  << variablesSize << "," << get_time << ","
                  << (end_step - start_step).count() / 1000 << std::endl;

        dataSpacesReader.Close();
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
