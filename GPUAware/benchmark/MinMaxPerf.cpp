#include <Kokkos_Core.hpp>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

double GPUMinMax(Kokkos::View<float *, Kokkos::DefaultExecutionSpace::memory_space> data, size_t size)
{
    auto tm_start = std::chrono::steady_clock::now();
    float min, max;
    Kokkos::parallel_reduce(
    size,
    KOKKOS_LAMBDA(int i, float &lmax, float &lmin) {
        if (lmax < data[i])
            lmax = data[i];
        if (lmin > data[i])
            lmin = data[i];
    },
    Kokkos::Max<float>(max), Kokkos::Min<float>(min));
    auto tm_end = std::chrono::steady_clock::now();
    return double((tm_end - tm_start).count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
}

double CPUMinMax(std::vector<float> data)
{
    auto tm_start = std::chrono::steady_clock::now();
    auto [min, max] = std::minmax_element(data.begin(), data.end());
    auto tm_end = std::chrono::steady_clock::now();
    return double((tm_end - tm_start).count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
}

int main(int argc, char **argv)
{
    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    std::uniform_int_distribution<int> dist {1, 52};

    auto gen = [&dist, &mersenne_engine](){
                   return dist(mersenne_engine);
               };

    Kokkos::initialize(argc, argv);
    using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;
    Kokkos::DefaultExecutionSpace exe_space;
    std::cout << "DefaultMemorySpace : " << exe_space.name() <<  std::endl;
    std::cout << "MinMax power(size=10^power) size(MB) timeKokkos(s) timeSTD(s)" << std::endl;
    for (size_t i = 6; i <= 8; i++)
    {
        size_t size = std::pow(10, i);
        std::vector<float> cpuData(size);
        std::generate(cpuData.begin(), cpuData.end(), gen);
        Kokkos::View<float *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> tempView(cpuData.data(), size);
        auto gpuData = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tempView);
        Kokkos::fence();

        auto timeGPU = GPUMinMax(gpuData, size);
        auto timeCPU = CPUMinMax(cpuData);
        std::cout << "MinMax " << i << " " << (size * sizeof(float)) / (1024 * 1024) << " "
                  << timeGPU << " " << timeCPU << std::endl;
    }

    Kokkos::finalize();
    return 0;
}
