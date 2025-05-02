#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

double GPUMagnitude(std::vector<float *> data, size_t size)
{
    Kokkos::fence();
    Kokkos::Timer timer;
    Kokkos::View<float *, Kokkos::DefaultExecutionSpace::memory_space> mag("Magnitude", size);
    std::cout << "GPU allocate " << timer.seconds();
    auto num_operands = data.size();
    Kokkos::View<float **, Kokkos::DefaultExecutionSpace::memory_space> data2("test", num_operands, size);
    parallel_for("ComputeMagnitude",
            Kokkos::MDRangePolicy< Kokkos::Rank<2> > ({0, 0}, {num_operands, size}),
            KOKKOS_LAMBDA(int i, int j) {
                auto val = data2(i, j);
                Kokkos::atomic_add(&mag[j], (val * val));
            }); 
    Kokkos::fence();
    parallel_for("ApplySqrt",
            Kokkos::RangePolicy<>(0, size),
            KOKKOS_LAMBDA(int i) {
                mag[i] = Kokkos::sqrt(mag[i]); 
            }); 
    Kokkos::fence();
    return timer.seconds();
}

double CPUMagnitude(std::vector<float *> data, size_t size)
{
    Kokkos::Timer timer;
    std::vector<float> mag(size, 0);
    for (auto operand : data){
        for (size_t i=0; i<size; i++){
            mag[i] += (operand[i] * operand[i]);
        }
    }
    return timer.seconds();
}

template <class T>
double GPUMinMax(T *data, size_t size)
{
    Kokkos::fence();
    Kokkos::Timer timer;
    T min, max;
    Kokkos::parallel_reduce(
    size,
    KOKKOS_LAMBDA(int i, T &lmax, T &lmin) {
        if (lmax < data[i])
            lmax = data[i];
        if (lmin > data[i])
            lmin = data[i];
    },
    Kokkos::Max<T>(max), Kokkos::Min<T>(min));
    Kokkos::fence();
    return timer.seconds();
}

template <class T>
double CPUMinMax(T *data_ptr, size_t size)
{
    Kokkos::Timer timer;
    std::vector<T> data(data_ptr, data_ptr + size);
    auto [min, max] = std::minmax_element(data.begin(), data.end());
    return timer.seconds();
}

template<class T>
class adiosGPU
{
    size_t size;
    std::vector<T *> inputData;
    public:
    adiosGPU(size_t data_size): size(data_size){};
    double Put(T *new_data)
    {
        inputData.push_back(new_data);
        auto timeMinMax = GPUMinMax(new_data, size);
        std::cout << "MinMax Device " << (size * sizeof(float)) / (1024 * 1024) << " "
                  << timeMinMax << " 1" << std::endl;
        auto timeMag = GPUMagnitude(inputData, size);
        std::cout << "Magnitude Device " << (size * sizeof(float)) / (1024 * 1024) << " "
                  << timeMag << " " << inputData.size() << std::endl;
        return timeMinMax + timeMag;
    }
};

template<class T>
class adiosCPU
{
    size_t size;
    std::vector<T *> inputData;
    public:
    adiosCPU(size_t data_size): size(data_size){};
    double Put(T *new_data)
    {
        inputData.push_back(new_data);
        auto timeMinMax = CPUMinMax(new_data, size);
        std::cout << "MinMax Host " << (size * sizeof(float)) / (1024 * 1024) << " "
                  << timeMinMax << " 1" << std::endl;
        auto timeMag = CPUMagnitude(inputData, size);
        std::cout << "Magnitude Host " << (size * sizeof(float)) / (1024 * 1024) << " "
                  << timeMag << " " << inputData.size() << std::endl;
        return timeMinMax + timeMag;
    }
};

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
    std::cout << "Kernel MemSpace size(MB) timeKokkos(s) NumberArrays" << std::endl;
    for (size_t i = 6; i <= 9; i++)
    {
        size_t size = std::pow(10, i);
        std::vector<float> cpuData(size);
        std::generate(cpuData.begin(), cpuData.end(), gen);
        Kokkos::View<float *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> tempView(cpuData.data(), size);
        auto gpuData = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tempView);
        Kokkos::fence();

        auto timeGPU = GPUMinMax(gpuData.data(), size);
        auto timeCPU = CPUMinMax(cpuData.data(), size);
        std::cout << "MinMax " << i << " " << (size * sizeof(float)) / (1024 * 1024) << " "
                  << timeGPU << " " << timeCPU << std::endl;
/*
        timeGPU = GPUMagnitude(gpuData, size);
        timeCPU = CPUMagnitude(cpuData);
        std::cout << "Magnitude " << i << " " << (size * sizeof(float)) / (1024 * 1024) << " "
                  << timeGPU << " " << timeCPU << std::endl;
*/    }

    Kokkos::finalize();
    return 0;
}
