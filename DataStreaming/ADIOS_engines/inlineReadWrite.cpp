#include <algorithm> //std::for_each
#include <ios>       //std::ios_base::failure
#include <iostream>  //std::cout
#include <stdexcept> //std::invalid_argument std::exception
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

void DoAnalysis(adios2::IO &inlineIO, adios2::Engine &inlineReader, int rank,
                int size, size_t variablesSize, unsigned int step)
{
    double get_time = 0;
    int Nx = 0;
    auto start_step = std::chrono::steady_clock::now();
    inlineReader.BeginStep();
    // READ
    for (unsigned int v = 0; v < variablesSize; ++v)
    {
        std::string namev("inlineFloats");
        namev += std::to_string(v);
        adios2::Variable<float> inlineFloats =
            inlineIO.InquireVariable<float>(namev);

        if (inlineFloats)
        {
            Nx = (inlineFloats.Shape()[0] / size);
            auto blocksInfo = inlineReader.BlocksInfo(inlineFloats, step);

            auto start_get = std::chrono::steady_clock::now();
            for (auto &info : blocksInfo)
            {
                // bp file reader would see all blocks, inline only sees local
                // writer's block(s).
                size_t myBlock = info.BlockID;
                inlineFloats.SetBlockSelection(myBlock);

                inlineReader.Get<float>(inlineFloats, info,
                                        adios2::Mode::Deferred);
            }
            inlineReader.PerformGets();
            auto end_get = std::chrono::steady_clock::now();
            get_time += (end_get - start_get).count() / 1000;
        }
        else
        {
            std::cout << "Variable inlineFloats not found\n";
        }
    }

    inlineReader.EndStep();
    auto end_step = std::chrono::steady_clock::now();
    double total_time = (end_step - start_step).count() / 1000;

    double global_get_sum;
    MPI_Reduce(&get_time, &global_get_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
           MPI_COMM_WORLD);
    double global_sum;
    MPI_Reduce(&total_time, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
           MPI_COMM_WORLD);

    // Time in microseconds
    if (rank == 0)
        std::cout << "Inline,Read," << size << "," << Nx << ","
                  << variablesSize << "," << global_get_sum / size << ","
                  << global_sum / size << std::endl;
    // all deferred block info are now valid - need data pointers to be
    // valid, filled with data
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

    // Application variable
    auto myFloats = create_random_data(Nx);

    try
    {
        // Inline uses single IO for write/read
        adios2::ADIOS adios(MPI_COMM_WORLD);
        adios2::IO inlineIO = adios.DeclareIO("InlineReadWrite");
        // WRITE
        inlineIO.SetEngine("Inline");

        std::vector<adios2::Variable<float>> inlineFloats(variablesSize);
        for (unsigned int v = 0; v < variablesSize; ++v)
        {
            std::string namev("inlineFloats");
            namev += std::to_string(v);
            inlineFloats[v] = inlineIO.DefineVariable<float>(
                namev, {size * Nx}, {rank * Nx}, {Nx}, adios2::ConstantDims);
        }

        adios2::Engine inlineWriter =
            inlineIO.Open("myWriteID", adios2::Mode::Write);

        adios2::Engine inlineReader =
            inlineIO.Open("myReadID", adios2::Mode::Read);

        for (unsigned int timeStep = 0; timeStep < 1; ++timeStep)
        {
            double put_time = 0;
            auto start_step = std::chrono::steady_clock::now();
            inlineWriter.BeginStep();
            for (unsigned int v = 0; v < variablesSize; ++v)
            {
                myFloats[rank] += static_cast<float>(v + rank);
                auto start_put = std::chrono::steady_clock::now();
                inlineWriter.Put(inlineFloats[v], myFloats.data());
                auto end_put = std::chrono::steady_clock::now();
                put_time += (end_put - start_put).count() / 1000;
            }
            inlineWriter.EndStep();
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
		std::cout << "Inline,Write," << size << "," << Nx << ","
			  << variablesSize << "," << global_put_sum / size << ","
			  << global_sum / size << std::endl;

            DoAnalysis(inlineIO, inlineReader, rank, size,
                       variablesSize, timeStep);
        }
    }
    catch (std::exception const &e)
    {
        std::cout << "Caught exception from rank " << rank << "\n";
        std::cout << e.what() << "\n";
        return MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
