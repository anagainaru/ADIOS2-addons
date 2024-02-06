#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

double GPUMagnitude(Kokkos::View<float *, Kokkos::DefaultExecutionSpace::memory_space> data, size_t size)
{
    Kokkos::fence();
    Kokkos::Timer timer;
    float mag = 0;
    Kokkos::parallel_reduce(
      size,
      KOKKOS_LAMBDA(int i, float &lmag) {
            lmag = data(i) * data(i);
      }, Kokkos::Sum<float>(mag));
    Kokkos::fence();
    return timer.seconds();
}

double CPUMagnitude(std::vector<float> data)
{
    Kokkos::Timer timer;
    float mag = std::accumulate(data.begin(), data.end(), 0, [](float a, float b){ return a + b * b;});
    return timer.seconds();
}

double GPUTranspose(Kokkos::View<float *, Kokkos::DefaultExecutionSpace::memory_space> data, size_t size)
{
    Kokkos::fence();
    Kokkos::Timer timer;
    Kokkos::View<float *, Kokkos::DefaultExecutionSpace::memory_space> dest("dest", size);
    Kokkos::Experimental::reverse_copy(Kokkos::DefaultExecutionSpace(), Kokkos::Experimental::begin(data), Kokkos::Experimental::end(data), Kokkos::Experimental::begin(dest));
    Kokkos::fence();
    return timer.seconds();
}

double CPUTranspose(std::vector<float> data, size_t size)
{
    Kokkos::Timer timer;
    std::vector<float> dest(size);
    std::reverse_copy(std::begin(data), std::end(data), std::begin(dest));
    return timer.seconds();
}

double GPUCopy(Kokkos::View<float *, Kokkos::DefaultExecutionSpace::memory_space> data, size_t size)
{
    Kokkos::fence();
    Kokkos::Timer timer;
    Kokkos::View<float *, Kokkos::HostSpace> dest("dest", size);
    Kokkos::deep_copy(dest, data);
    Kokkos::fence();
    return timer.seconds();
}

double CPUCopy(std::vector<float> data, size_t size)
{
    Kokkos::Timer timer;
    std::vector<float> dest(size);
    std::copy(data.begin(), data.end(), dest.begin());
    return timer.seconds();
}

double GPUMinMax(Kokkos::View<float *, Kokkos::DefaultExecutionSpace::memory_space> data, size_t size)
{
    Kokkos::fence();
    Kokkos::Timer timer;
    float min, max;
    Kokkos::parallel_reduce(
    size,
    KOKKOS_LAMBDA(int i, float &lmax, float &lmin) {
        if (lmax < data(i))
            lmax = data(i);
        if (lmin > data(i))
            lmin = data(i);
    },
    Kokkos::Max<float>(max), Kokkos::Min<float>(min));
    Kokkos::fence();
    return timer.seconds();
}

double CPUMinMax(std::vector<float> data)
{
    Kokkos::Timer timer;
    auto [min, max] = std::minmax_element(data.begin(), data.end());
    return timer.seconds();
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
    std::cout << "Kernel power(size=10^power) size(MB) timeKokkos(s) timeSTD(s)" << std::endl;
    for (size_t i = 6; i <= 9; i++)
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

        timeGPU = GPUCopy(gpuData, size);
        timeCPU = CPUCopy(cpuData, size);
        std::cout << "Copy " << i << " " << (size * sizeof(float)) / (1024 * 1024) << " "
                  << timeGPU << " " << timeCPU << std::endl;

        timeGPU = GPUTranspose(gpuData, size);
        timeCPU = CPUTranspose(cpuData, size);
        std::cout << "Transpose " << i << " " << (size * sizeof(float)) / (1024 * 1024) << " "
                  << timeGPU << " " << timeCPU << std::endl;

        timeGPU = GPUMagnitude(gpuData, size);
        timeCPU = CPUMagnitude(cpuData);
        std::cout << "Magnitude " << i << " " << (size * sizeof(float)) / (1024 * 1024) << " "
                  << timeGPU << " " << timeCPU << std::endl;
    }

    Kokkos::finalize();
    return 0;
}
