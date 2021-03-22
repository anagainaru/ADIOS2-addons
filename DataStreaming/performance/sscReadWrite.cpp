#include <iostream>
#include <vector>
#include <chrono>
#include <random>

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

void read_data(adios2::IO sscIO, int rank, int size,
               size_t variablesSize)
{
    double get_time = 0;
    size_t Nx = 0;
    std::vector<std::vector<float>> myFloats(variablesSize);
    adios2::Engine sscReader = sscIO.Open("helloSsc", adios2::Mode::Read);
    auto start_step = std::chrono::steady_clock::now();
    sscReader.LockReaderSelections();
    sscReader.BeginStep();
    for (unsigned int v = 0; v < variablesSize; v++)
    {
        std::string namev("sscFloats");
        namev += std::to_string(v);
        adios2::Variable<float> sscFloats =
            sscIO.InquireVariable<float>(namev);

        const std::size_t total_size = sscFloats.Shape()[0];
        const std::size_t my_start = (total_size / size) * rank;
        const std::size_t my_count = (total_size / size);
        Nx = my_count;

        const adios2::Dims start{my_start};
        const adios2::Dims count{my_count};

        const adios2::Box<adios2::Dims> sel(start, count);

        myFloats[v].resize(my_count);
        sscFloats.SetSelection(sel);
        auto start_get = std::chrono::steady_clock::now();
        sscReader.Get(sscFloats, myFloats[v].data());
        auto end_get = std::chrono::steady_clock::now();
        get_time += (end_get - start_get).count() / 1000;
    }
    sscReader.EndStep();
    auto end_step = std::chrono::steady_clock::now();
    // Time in microseconds
    std::cout << "SSC,Read," << rank << ","  << Nx << ","
              << variablesSize << "," << get_time << ","
              << (end_step - start_step).count() / 1000 << std::endl;
    sscReader.Close();
}

void write_data(adios2::IO sscIO, int rank, int size,
                size_t Nx, size_t variablesSize)
{
    // Application variable
    auto myFloats = create_random_data(Nx);
    adios2::Engine sscWriter = sscIO.Open("helloSsc", adios2::Mode::Write);

    // Define variable and local size
    std::vector<adios2::Variable<float>> sscFloats(variablesSize);
    for (unsigned int v = 0; v < variablesSize; ++v)
    {
        std::string namev("sscFloats");
        namev += std::to_string(v);
        sscFloats[v] = sscIO.DefineVariable<float>(
            namev, {size * Nx}, {rank * Nx}, {Nx});
    }

    double put_time = 0;
    auto start_step = std::chrono::steady_clock::now();
    sscWriter.LockWriterDefinitions();
    sscWriter.BeginStep();
    for (unsigned int v = 0; v < variablesSize; v++)
    {
        myFloats[rank] += static_cast<float>(v + rank);
        auto start_put = std::chrono::steady_clock::now();
        sscWriter.Put<float>(sscFloats[v], myFloats.data());
        auto end_put = std::chrono::steady_clock::now();
        put_time += (end_put - start_put).count() / 1000;
    }
    sscWriter.EndStep();
    auto end_step = std::chrono::steady_clock::now();
    // Time in microseconds
    std::cout << "SSC,Write," << rank << ","  << Nx << ","
              << variablesSize << "," << put_time << ","
              << (end_step - start_step).count() / 1000 << std::endl;
    sscWriter.Close();
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm mpiComm;
    int worldRank, worldSize;
    int rank, size;
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " array_size number_variables"
                  << std::endl;
        return -1;
    }
    const size_t Nx = atoi(argv[1]);
    const size_t variablesSize = atoi(argv[2]);

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
            write_data(sscIO, rank, size, Nx, variablesSize);
        if (mpiGroup == 1)
            read_data(sscIO, rank, size, variablesSize);
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
