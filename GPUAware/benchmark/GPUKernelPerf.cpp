#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

double GPUAdd(
    Kokkos::View<double **, Kokkos::DefaultExecutionSpace::memory_space> data,
    Kokkos::View<double *, Kokkos::DefaultExecutionSpace::memory_space> &dest) {
  Kokkos::fence();
  Kokkos::Timer timer;
  double mag = 0;
  int size = data.extent(0);
  int numVar = data.extent(1);
  Kokkos::parallel_for("add", size, KOKKOS_LAMBDA(int i) {
    dest(i) = 0;
    for (int j = 0; j < numVar; j++) {
      dest(i) += data(i, j);
    }
  });
  Kokkos::fence();
  return timer.seconds();
}

double CPUAdd(std::vector<std::vector<double>> inputData,
              std::vector<double> &outValues) {
  Kokkos::Timer timer;
  size_t dataSize = inputData[0].size();
  for (auto &variable : inputData) {
    for (size_t i = 0; i < dataSize; i++) {
      outValues[i] = outValues[i] + variable[i];
    }
  }
  return timer.seconds();
}

double GPUMagnitude(
    Kokkos::View<double **, Kokkos::DefaultExecutionSpace::memory_space> data,
    Kokkos::View<double *, Kokkos::DefaultExecutionSpace::memory_space> &dest) {
  Kokkos::fence();
  Kokkos::Timer timer;
  double mag = 0;
  int size = data.extent(0);
  int numVar = data.extent(1);
  Kokkos::parallel_for("magnitude", size, KOKKOS_LAMBDA(int i) {
    dest(i) = 0;
    for (int j = 0; j < size; j++) {
      dest(i) += data(i, j) * data(i, j);
    }
  });
  Kokkos::fence();
  return timer.seconds();
}

double CPUMagnitude(std::vector<std::vector<double>> inputData,
                    std::vector<double> &outValues) {
  Kokkos::Timer timer;
  size_t dataSize = inputData[0].size();
  for (auto &variable : inputData) {
    for (size_t i = 0; i < dataSize; i++) {
      outValues[i] = outValues[i] + variable[i] * variable[i];
    }
  }
  for (size_t i = 0; i < dataSize; i++) {
    outValues[i] = std::sqrt(outValues[i]);
  }
  return timer.seconds();
}

double GPUCurl(
    Kokkos::View<double ***, Kokkos::DefaultExecutionSpace::memory_space> data1,
    Kokkos::View<double ***, Kokkos::DefaultExecutionSpace::memory_space> data2,
    Kokkos::View<double ***, Kokkos::DefaultExecutionSpace::memory_space> data3,
    Kokkos::View<double ***[3], Kokkos::DefaultExecutionSpace::memory_space> &dest)
{
  Kokkos::fence();
  Kokkos::Timer timer;
  int dimx = dest.extent(0);
  int dimy = dest.extent(1);
  int dimz = dest.extent(2);
  Kokkos::parallel_for(
      "compute_curl",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {dimx, dimy, dimz}),
      KOKKOS_LAMBDA(int i, int j, int k) {
        int next_i = std::max(0, i - 1), prev_i = std::min(dimx - 1, i + 1);
        int next_j = std::max(0, j - 1), prev_j = std::min(dimy - 1, j + 1);
        int next_k = std::max(0, k - 1), prev_k = std::min(dimz - 1, k + 1);
        // curl[0] = dv3 / dy - dv2 / dz
        dest(i, j, k, 0) =
            (data3(i, next_j, k) - data3(i, prev_j, k)) / (next_j - prev_j);
        dest(i, j, k, 0) +=
            (data2(i, j, prev_k) - data2(i, j, next_k)) / (next_k - prev_k);
        // curl[1] = dv1 / dz - dv3 / dx
        dest(i, j, k, 1) =
            (data1(i, j, next_k) - data1(i, j, prev_k)) / (next_k - prev_k);
        dest(i, j, k, 1) +=
            (data3(prev_i, j, k) - data3(next_i, j, k)) / (next_i - prev_i);
        // curl[2] = dv2 / dx - dv1 / dy
        dest(i, j, k, 2) =
            (data2(next_i, j, k) - data2(prev_i, j, k)) / (next_i - prev_i);
        dest(i, j, k, 2) +=
            (data1(i, prev_j, k) - data1(i, next_j, k)) / (next_j - prev_j);
      });

  Kokkos::fence();
  return timer.seconds();
}

inline size_t returnIndex(size_t x, size_t y, size_t z, size_t dims[3]) {
  return z + y * dims[2] + x * dims[2] * dims[1];
}

double CPUCurl(std::vector<double> inputData[3], size_t dims[3],
               std::vector<double> &outValues) {
  Kokkos::Timer timer;
  size_t index = 0;
  for (size_t k = 0; k < dims[2]; ++k) {
    size_t next_k = std::max((size_t)0, k - 1),
           prev_k = std::min(dims[2] - 1, k + 1);
    for (size_t j = 0; j < dims[1]; ++j) {
      size_t next_j = std::max((size_t)0, j - 1),
             prev_j = std::min(dims[1] - 1, j + 1);
      for (size_t i = 0; i < dims[0]; ++i) {
        size_t next_i = std::max((size_t)0, i - 1),
               prev_i = std::min(dims[0] - 1, i + 1);
        // curl[0] = dv2 / dy - dv1 / dz
        outValues[3 * index] = (inputData[2][returnIndex(i, next_j, k, dims)] -
                                inputData[2][returnIndex(i, prev_j, k, dims)]) /
                               (next_j - prev_j);
        outValues[3 * index] +=
            (inputData[1][returnIndex(i, j, prev_k, dims)] -
             inputData[1][returnIndex(i, j, next_k, dims)]) /
            (next_k - prev_k);
        // curl[1] = dv0 / dz - dv2 / dx
        outValues[3 * index + 1] =
            (inputData[0][returnIndex(i, j, next_k, dims)] -
             inputData[0][returnIndex(i, j, prev_k, dims)]) /
            (next_k - prev_k);
        outValues[3 * index + 1] +=
            (inputData[2][returnIndex(prev_i, j, k, dims)] -
             inputData[2][returnIndex(next_i, j, k, dims)]) /
            (next_i - prev_i);
        // curl[2] = dv1 / dx - dv0 / dy
        outValues[3 * index + 2] =
            (inputData[1][returnIndex(next_i, j, k, dims)] -
             inputData[1][returnIndex(prev_i, j, k, dims)]) /
            (next_i - prev_i);
        outValues[3 * index + 2] +=
            (inputData[0][returnIndex(i, prev_j, k, dims)] -
             inputData[0][returnIndex(i, next_j, k, dims)]) /
            (next_j - prev_j);
        index++;
      }
    }
  }
  return timer.seconds();
}

double GPUCopy(
    Kokkos::View<double *, Kokkos::DefaultExecutionSpace::memory_space> data,
    Kokkos::View<double *, Kokkos::HostSpace> &dest) {
  Kokkos::fence();
  Kokkos::Timer timer;
  Kokkos::deep_copy(dest, data);
  Kokkos::fence();
  return timer.seconds();
}

double CPUCopy(std::vector<double> data, std::vector<double> &dest) {
  Kokkos::Timer timer;
  std::copy(data.begin(), data.end(), dest.begin());
  return timer.seconds();
}

double GPUMinMax(
    Kokkos::View<double *, Kokkos::DefaultExecutionSpace::memory_space> data,
    size_t size, double &ret) {
  Kokkos::fence();
  Kokkos::Timer timer;
  double min, max;
  Kokkos::parallel_reduce(size,
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

double CPUMinMax(std::vector<double> data, double &ret) {
  Kokkos::Timer timer;
  auto [min, max] = std::minmax_element(data.begin(), data.end());
  ret = *min + *max;
  return timer.seconds();
}

void default_tests(size_t size) {
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<int> dist{1, 52};

  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
  for (size_t i = 1; i < 4; i++) {
    std::vector<double> cpuData(size);
    std::generate(cpuData.begin(), cpuData.end(), gen);
    Kokkos::View<double *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        tempView(cpuData.data(), size);
    auto gpuData = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace(), tempView);
    Kokkos::fence();

    double statsGPU, statsCPU;
    auto timeGPU = GPUMinMax(gpuData, size, statsGPU);
    auto timeCPU = CPUMinMax(cpuData, statsCPU);
    if (statsGPU != statsCPU)
      std::cout << "DEBUG Error: value mismatch in minmax" << std::endl;
    std::cout << "MinMax " << i << " "
              << (size * sizeof(double)) / (1024 * 1024) << " " << timeGPU
              << " " << timeCPU << std::endl;

    std::vector<double> destCPU(size);
    Kokkos::View<double *, Kokkos::HostSpace> destGPU("dest", size);
    timeGPU = GPUCopy(gpuData, destGPU);
    timeCPU = CPUCopy(cpuData, destCPU);
    if (destGPU[0] != destCPU[0])
      std::cout << "DEBUG Error: value mismatch in copy" << std::endl;
    std::cout << "Copy " << i << " " << (size * sizeof(double)) / (1024 * 1024)
              << " " << timeGPU << " " << timeCPU << std::endl;
  }
}

void derived_tests(size_t size, size_t numVar) {
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<int> dist{1, 52};

  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);

  for (size_t i = 1; i < 4; i++) {
    std::vector<std::vector<double>> cpuList;
    int sumElem = 0, magElem = 0;
    for (int var = 0; var < numVar; var++) {
      std::vector<double> cpuData(size);
      std::generate(cpuData.begin(), cpuData.end(), gen);
      cpuList.push_back(cpuData);
      sumElem += cpuData[0];
      magElem += (cpuData[0] * cpuData[0]);
    }
    magElem = std::sqrt(magElem);

    Kokkos::View<double **, Kokkos::DefaultExecutionSpace::memory_space>
        gpuList("derivedBuf", size, numVar);
    Kokkos::parallel_for("initialize", size, KOKKOS_LAMBDA(int i) {
      auto generator = random_pool.get_state();
      for (int j = 0; j < numVar; i++) {
        gpuList(i, j) = generator.drand(0., 1.);
      }
      random_pool.free_state(generator);
    });

    std::vector<double> destCPU(size);
    Kokkos::View<double *, Kokkos::DefaultExecutionSpace::memory_space> destGPU(
        "derivedDest", size);
    auto timeGPU = GPUAdd(gpuList, destGPU);
    auto timeCPU = CPUAdd(cpuList, destCPU);
    if (sumElem != destCPU[0])
      std::cout << "DEBUG Error: value mismatch in add" << std::endl;
    std::cout << "Add " << i << " " << (size * sizeof(double)) / (1024 * 1024)
              << " " << timeGPU << " " << timeCPU << std::endl;

    timeGPU = GPUMagnitude(gpuList, destGPU);
    timeCPU = CPUMagnitude(cpuList, destCPU);
    if (magElem != destCPU[0])
      std::cout << "DEBUG Error: value mismatch in magnitude" << std::endl;
    std::cout << "Magnitude " << i << " "
              << (size * sizeof(double)) / (1024 * 1024) << " " << timeGPU
              << " " << timeCPU << std::endl;
  }
}

void curl_tests(size_t dimx, size_t dimy, size_t dimz) {
  size_t dims[3] = {dimx, dimy, dimz};
  size_t size = dimx * dimy * dimz;
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<int> dist{1, 52};

  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);

  for (size_t i = 1; i < 4; i++) {
    std::vector<double> cpuData1(size);
    std::vector<double> cpuData2(size);
    std::vector<double> cpuData3(size);
    std::generate(cpuData1.begin(), cpuData1.end(), gen);
    std::generate(cpuData2.begin(), cpuData2.end(), gen);
    std::generate(cpuData3.begin(), cpuData3.end(), gen);
    std::vector<double> cpuList[3] = {cpuData1, cpuData2, cpuData3};
    std::vector<double> destCPU(3 * size);

    Kokkos::View<double ***, Kokkos::DefaultExecutionSpace::memory_space>
        gpuData1("derivedBuf1", dims[0], dims[1], dims[2]);
    Kokkos::View<double ***, Kokkos::DefaultExecutionSpace::memory_space>
        gpuData2("derivedBuf2", dims[0], dims[1], dims[2]);
    Kokkos::View<double ***, Kokkos::DefaultExecutionSpace::memory_space>
        gpuData3("derivedBuf3", dims[0], dims[1], dims[2]);
    Kokkos::parallel_for("init_buffers",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                             {0, 0, 0}, {dims[0], dims[1], dims[2]}),
                         KOKKOS_LAMBDA(int i, int j, int k) {
                           auto generator = random_pool.get_state();
                           gpuData1(i, j, k) = generator.drand(0., 1.);
                           gpuData2(i, j, k) = generator.drand(0., 1.);
                           gpuData3(i, j, k) = generator.drand(0., 1.);
                           random_pool.free_state(generator);
                         });
    Kokkos::View<double ** * [3], Kokkos::DefaultExecutionSpace::memory_space>
        destGPU("curlDest", dims[0], dims[1], dims[2]);
    auto timeGPU = GPUCurl(gpuData1, gpuData2, gpuData3, destGPU);
    auto timeCPU = CPUCurl(cpuList, dims, destCPU);
    std::cout << "DEBUG First curl value " << destCPU[0] << std::endl;
    std::cout << "Curl " << i << " " << (size * sizeof(double)) / (1024 * 1024)
              << " " << timeGPU << " " << timeCPU << dims[0] << dims[1]
              << dims[2] << std::endl;
  }
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;
  Kokkos::DefaultExecutionSpace exe_space;
  std::cout << "DEBUG DefaultMemorySpace : " << exe_space.name() << std::endl;
  std::vector<size_t> dim_list = {254, 309, 320, 358, 366, 374, 382, 389, 403};
  std::vector<size_t> variable_list = {2}; //, 10, 100};
  for (const size_t &dim : dim_list) {
    size_t size = dim * dim * dim;
    std::cout << "DEBUG Start default tests" << std::endl;
    default_tests(size);
    std::cout << "DEBUG End default tests" << std::endl;
    std::cout << "DEBUG Start derived tests" << std::endl;
    for (const size_t &numVar : variable_list)
      derived_tests(size, numVar);

    curl_tests(dim, dim, dim);
    std::cout << "DEBUG End derived tests" << std::endl;
  }

  Kokkos::finalize();
  return 0;
}
