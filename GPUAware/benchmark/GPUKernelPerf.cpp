#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <Kokkos_Random.hpp>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>


double GPUAdd(Kokkos::View<double **> data,
              Kokkos::View<double *> dest)
{
    Kokkos::fence();
    Kokkos::Timer timer;
    double mag = 0;
    int size = data.extent(0);
    int numVar = data.extent(1);
    Kokkos::parallel_for(
      "add", size,
      KOKKOS_LAMBDA(int i) {
            dest(i) = 0;
            for (int j=0; j<numVar; j++)
            {
                dest(i) += data(i, j);
            }
      });
    Kokkos::fence();
    return timer.seconds();
}

double CPUAdd(std::vector<std::vector<double>> inputData, std::vector<double> outValues)
{
    Kokkos::Timer timer;
    size_t dataSize = inputData[0].size();
    for (auto &variable : inputData)
    {
        for (size_t i = 0; i < dataSize; i++)
        {
            outValues[i] = outValues[i] + variable[i];
        }
    }
    return timer.seconds();
}

double GPUMagnitude(Kokkos::View<double **> data,
                    Kokkos::View<double *> dest)
{
    Kokkos::fence();
    Kokkos::Timer timer;
    double mag = 0;
    int size = data.extent(0);
    int numVar = data.extent(1);
    Kokkos::parallel_for(
      "magnitude", size,
      KOKKOS_LAMBDA(int i) {
            dest(i) = 0;
            for (int j=0; j<numVar; j++)
            {
                dest(i) += data(i, j) * data(i, j);
            }
      });
    Kokkos::fence();
    return timer.seconds();
}

double CPUMagnitude(std::vector<std::vector<double>> inputData, std::vector<double> outValues)
{
    Kokkos::Timer timer;
    size_t dataSize = inputData[0].size();
    for (auto &variable : inputData)
    {
        for (size_t i = 0; i < dataSize; i++)
        {
            outValues[i] = outValues[i] + variable[i] * variable[i];
        }
    }
    for (size_t i = 0; i < dataSize; i++)
    {
        outValues[i] = std::sqrt(outValues[i]);
    }
    return timer.seconds();
}

double GPUCopy(Kokkos::View<double *> data,
               Kokkos::View<double *, Kokkos::HostSpace> dest)
{
    Kokkos::fence();
    Kokkos::Timer timer;
    Kokkos::deep_copy(dest, data);
    Kokkos::fence();
    return timer.seconds();
}

double CPUCopy(std::vector<double> data, std::vector<double> dest)
{
    Kokkos::Timer timer;
    std::copy(data.begin(), data.end(), dest.begin());
    return timer.seconds();
}

double GPUMinMax(Kokkos::View<double *> data,
                 size_t size, double &ret)
{
    Kokkos::fence();
    Kokkos::Timer timer;
    double min, max;
    Kokkos::parallel_reduce(
    size,
    KOKKOS_LAMBDA(int i, double &lmax, double &lmin) {
        if (lmax < data(i))
            lmax = data(i);
        if (lmin > data(i))
            lmin = data(i);
    },
    Kokkos::Max<double>(max), Kokkos::Min<double>(min));
    Kokkos::fence();
    ret = min + max;
    return timer.seconds();
}

double CPUMinMax(std::vector<double> data, double &ret)
{
    Kokkos::Timer timer;
    auto [min, max] = std::minmax_element(data.begin(), data.end());
    ret = *min + *max;
    return timer.seconds();
}

void default_tests(size_t size)
{
    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    std::uniform_int_distribution<int> dist {1, 52};

    auto gen = [&dist, &mersenne_engine](){
                   return dist(mersenne_engine);
               };
    for(size_t i=1; i<4; i++)
    {
        std::vector<double> cpuData(size);
        std::generate(cpuData.begin(), cpuData.end(), gen);
        Kokkos::View<double *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> tempView(cpuData.data(), size);
        auto gpuData = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tempView);
        Kokkos::fence();

        double statsGPU, statsCPU;
        auto timeGPU = GPUMinMax(gpuData, size, statsGPU);
        auto timeCPU = CPUMinMax(cpuData, statsCPU);
        if (statsGPU != statsCPU) std::cout << "ERROR value mismatch in minmax" << std::endl;
        std::cout << "MinMax " << i << " " << (size * sizeof(double)) / (1024 * 1024) << " "
                  << timeGPU << " " << timeCPU << std::endl;

        std::vector<double> destCPU(size);
        Kokkos::View<double *, Kokkos::HostSpace> destGPU("dest", size);
        timeGPU = GPUCopy(gpuData, destGPU);
        timeCPU = CPUCopy(cpuData, destCPU);
        if (destGPU[0] != destCPU[0]) std::cout << "ERROR value mismatch in copy" << std::endl;
        std::cout << "Copy " << i << " " << (size * sizeof(double)) / (1024 * 1024) << " "
                  << timeGPU << " " << timeCPU << std::endl;
    }
}

void derived_tests(size_t size, size_t numVar)
{
    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    std::uniform_int_distribution<int> dist {1, 52};

    auto gen = [&dist, &mersenne_engine](){
                   return dist(mersenne_engine);
               };
    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);

    for(size_t i=1; i<4; i++)
    {
        std::vector<std::vector<double>> cpuList;
        int sumElem = 0, magElem = 0;
        for (int var=0; var<numVar; var++)
        {
            std::vector<double> cpuData(size);
            std::generate(cpuData.begin(), cpuData.end(), gen);
            cpuList.push_back(cpuData);
            sumElem += cpuData[0];
            magElem += (cpuData[0] * cpuData[0]);
        }
        Kokkos::View<double **> gpuList("derivedBuf", size, numVar);
        Kokkos::parallel_for(
        "initialize", size,
        KOKKOS_LAMBDA(int i) {
            auto generator = random_pool.get_state();
            for (int j=0; j<numVar; j++)
            {
                gpuList(i, j) = generator.drand(0., 1.);
            }
            random_pool.free_state(generator);
        });

        std::vector<double> destCPU(size);
        Kokkos::View<double *> destGPU("derivedDest", size);
        auto timeGPU = GPUAdd(gpuList, destGPU);
        auto timeCPU = CPUAdd(cpuList, destCPU);
        if (sumElem != destCPU[0]) std::cout << "ERROR value mismatch in add" << std::endl;
        std::cout << "Add " << i << " " << (size * sizeof(double)) / (1024 * 1024) << " "
                  << timeGPU << " " << timeCPU << " numvar " << numVar << std::endl;

        timeGPU = GPUMagnitude(gpuList, destGPU);
        timeCPU = CPUMagnitude(cpuList, destCPU);
        if (magElem != destCPU[0]) std::cout << "ERROR value mismatch in magnitude" << std::endl;
        std::cout << "Magnitude " << i << " " << (size * sizeof(double)) / (1024 * 1024) << " "
                  << timeGPU << " " << timeCPU << " numvar " << numVar << std::endl;
    }
}

int main(int argc, char **argv)
{
    Kokkos::initialize(argc, argv);
    using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;
    Kokkos::DefaultExecutionSpace exe_space;
    std::cout << "DefaultMemorySpace : " << exe_space.name() <<  std::endl;
    std::vector<size_t> size_list = {50*1048576};
    std::vector<size_t> variable_list = {2, 4, 8, 16};
    for (const size_t& size:size_list)
    {
        //default_tests(size);
        for (const size_t& numVar:variable_list)
            derived_tests(size, numVar);
    }

    Kokkos::finalize();
    return 0;
}
