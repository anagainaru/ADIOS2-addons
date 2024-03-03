#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

double GPUMagnitude(Kokkos::View<double *, Kokkos::DefaultExecutionSpace::memory_space> data, size_t size)
{
    Kokkos::fence();
    Kokkos::Timer timer;
    double mag = 0;
    Kokkos::parallel_reduce(
      size,
      KOKKOS_LAMBDA(int i, double &lmag) {
            lmag = data(i) * data(i);
      }, Kokkos::Sum<double>(mag));
    Kokkos::fence();
    return timer.seconds();
}

double CPUMagnitude(std::vector<double> data)
{
    Kokkos::Timer timer;
    double mag = std::accumulate(data.begin(), data.end(), 0, [](double a, double b){ return a + b * b;});
    return timer.seconds();
}

double GPUTranspose(Kokkos::View<double *, Kokkos::DefaultExecutionSpace::memory_space> data, size_t size)
{
    Kokkos::fence();
    Kokkos::Timer timer;
    Kokkos::View<double *, Kokkos::DefaultExecutionSpace::memory_space> dest("dest", size);
    Kokkos::Experimental::reverse_copy(Kokkos::DefaultExecutionSpace(), Kokkos::Experimental::begin(data), Kokkos::Experimental::end(data), Kokkos::Experimental::begin(dest));
    Kokkos::fence();
    return timer.seconds();
}

double CPUTranspose(std::vector<double> data, size_t size)
{
    Kokkos::Timer timer;
    std::vector<double> dest(size);
    std::reverse_copy(std::begin(data), std::end(data), std::begin(dest));
    return timer.seconds();
}

double GPUCopy(Kokkos::View<double *, Kokkos::DefaultExecutionSpace::memory_space> data,
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

double GPUMinMax(Kokkos::View<double *, Kokkos::DefaultExecutionSpace::memory_space> data,
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
	std::vector<size_t> size_list = {25*1048576, 50*1048576}; // {100*1048576, 200*1048576, 400*1048576};
    for (const size_t& size:size_list)
    {
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

	/*        timeGPU = GPUTranspose(gpuData, size);
			timeCPU = CPUTranspose(cpuData, size);
			std::cout << "Transpose " << i << " " << (size * sizeof(double)) / (1024 * 1024) << " "
					  << timeGPU << " " << timeCPU << std::endl;

			timeGPU = GPUMagnitude(gpuData, size);
			timeCPU = CPUMagnitude(cpuData);
			std::cout << "Magnitude " << i << " " << (size * sizeof(double)) / (1024 * 1024) << " "
					  << timeGPU << " " << timeCPU << std::endl;
	*/
		}
	}

    Kokkos::finalize();
    return 0;
}
