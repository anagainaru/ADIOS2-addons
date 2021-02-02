/*
 * Simple example of writing and reading data
 * through ADIOS2 BP engine with multiple simulations steps
 * for every IO step.
 */

#include <ios>
#include <vector>

#include <adios2.h>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <chrono>
#include <string>

#include "cufile.h"

int write_step = 1; // write every x-th simulation step

int WriteSim(const std::string fname, const size_t N,
            int nSteps, bool gpu_direct){
  // Initialize the simulation data
  Kokkos::View<float*> gpuSimData("simBuffer", N);
 
  // Set up the ADIOS structures
  adios2::ADIOS adios;
  adios2::IO io = adios.DeclareIO("WriteIO");

  const adios2::Dims shape{static_cast<size_t>(N)};
  const adios2::Dims start{static_cast<size_t>(0)};
  const adios2::Dims count{N};
  auto data = io.DefineVariable<float>("data", shape, start, count);

  adios2::Engine bpWriter = io.Open(fname, adios2::Mode::Write);

  std::cout << "Write data with ADIOS ";
  if (gpu_direct)
      std::cout << "directly from GPU buffers" << std::endl;
  else
      std::cout << "moving data to CPU space" << std::endl;

  auto start_time = std::chrono::steady_clock::now();
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_time;
  double elapsed_seconds = 0;
  // Simulation steps
  for (size_t step = 0; step < nSteps; ++step)
  {
      adios2::Box<adios2::Dims> sel({0}, {N});
      data.SetSelection(sel);

      // Start IO step every write_step
      if (step % write_step == 0){
	start_time = std::chrono::steady_clock::now();
        bpWriter.BeginStep();
	// either send directly the gpu buffer
	if (gpu_direct)
	    bpWriter.Put(data, gpuSimData.data());
	else
	{
	    // or copy the buffer to the cpu and then send
	    auto cpu_buf = Kokkos::create_mirror_view_and_copy(
          	Kokkos::HostSpace{}, gpuSimData);
	    bpWriter.Put(data, cpu_buf.data());
	}
	bpWriter.EndStep();
	end_time = std::chrono::steady_clock::now();
	elapsed_time = end_time - start_time;
	elapsed_seconds += elapsed_time.count();
      }

      // update the simulation buffer
      Kokkos::parallel_for("updateBuffer", N, KOKKOS_LAMBDA(int i){
	gpuSimData(i) += 5;
      });
  }

  if (gpu_direct)
      std::cout << "GDS write time: ";
  else
      std::cout << "CopyCPU write time: ";
  std::cout << elapsed_seconds << " N " << N << " steps "
	    << nSteps << std::endl;
  bpWriter.Close();
  return 0;
}

int BPRead(const std::string fname, const size_t N, int nSteps){
  std::cout << "Reading ADIOS files" << std::endl;

  // Create ADIOS structures
  adios2::ADIOS adios;
  adios2::IO io = adios.DeclareIO("ReadIO");

  adios2::Engine bpReader = io.Open(fname, adios2::Mode::Read);

  auto data = io.InquireVariable<float>("data");
  //std::cout << "Steps expected by the reader: " << bpReader.Steps() << std::endl;
  //std::cout << "Expecting data per step: " << data.Shape()[0];
  //std::cout  << " elements" << std::endl;

  int adios_steps = bpReader.Steps();
  // Create the local buffer and initialize the access point in the ADIOS file
  std::vector<float> simData(N); //set size to N
  const adios2::Dims start{0};
  const adios2::Dims count{N};
  const adios2::Box<adios2::Dims> sel(start, count);
  data.SetSelection(sel);
  
  // Read the data in each of the ADIOS steps
  for (size_t step = 0; step < adios_steps; step++)
  {
      data.SetStepSelection({step, 1});
      bpReader.Get(data, simData.data());
      bpReader.PerformGets();
      if (step == 0 || step == adios_steps - 1)
      {
          std::cout << "Simualation step " << step << " : ";
          std::cout << simData.size() << " elements: " << simData[1] << std::endl;
      }
  }
  bpReader.Close();
  return 0;
}

int GPURead(const std::string fname, const size_t N, int nSteps)
{
  Kokkos::View<float *> kokkos_buf( "gpu_buffer", N);

  std::cout << "Reading GPU file" << std::endl;
  int fd = open(fname.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) {
    std::cerr << "read file open error : " << fname << " "
        << std::strerror(errno) << std::endl;
    return -1;
  }

  int ret = -1;
  CUfileError_t status;
  CUfileHandle_t fh;
  CUfileDescr_t desc;
  memset((void *)&desc, 0, sizeof(CUfileDescr_t));

  desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  desc.handle.fd = fd;
  status = cuFileHandleRegister(&fh, &desc);
  if (status.err != CU_FILE_SUCCESS) {
    std::cerr << "file register error " << std::endl;
    close(fd);
    return -1;
  }

  int idx;
  cudaGetDevice(&idx);
  
  // Simulation steps
  for (size_t step = 0; step < nSteps / write_step; step++)
  {
      ret = cuFileRead(fh, kokkos_buf.data(), N * sizeof(float),
		       step * N * sizeof(float), 0);
      if (ret < 0) {
        std::cerr << "read failed at step " << step << std::endl;
        cuFileHandleDeregister(fh);
        close(fd);
        return -1;
      }
      auto cpu_buf = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{}, kokkos_buf);
      if (step == 0 || step == nSteps / write_step - 1)
      {
          std::cout << "Simulation step " << step << " " << N
      	   	    << " elements : " << cpu_buf[0] << std::endl;
      }
  }

  cuFileHandleDeregister(fh);
  close (fd);
  return 0;
}

int main(int argc, char **argv){
  if (argc < 3)
  {
      std::cout << "Usage: " << argv[0] << " array_size simulation_steps"
	        << std::endl;
      return -1;
  }
  const size_t N = atoi(argv[1]);
  int nSteps = atoi(argv[2]);

  int device_id = 1;
  Kokkos::InitArguments args;
  args.device_id = device_id;
  Kokkos::ScopeGuard gds(args);

  // ADIOS folder
  const std::string fname("/mnt/nvme/KokkosWrite.bp");
  // simulation steps, array size
  int ret = 0;

  auto start = std::chrono::steady_clock::now();
  ret = WriteSim(fname, N, nSteps, true);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "GDS total simulation time: " << elapsed_seconds.count();
  std::cout << " N " << N << " steps " << nSteps << std::endl;
  std::remove((fname + "/gpu.0.1").c_str());
  // if we want to read the data and print the first/last element:
  //  ret += GPURead(fname + "/gpu.0." + std::to_string(device_id), N, nSteps);
  
  start = std::chrono::steady_clock::now();
  ret += WriteSim(fname, N, nSteps, false);
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end-start;
  std::cout << "CopyCPU total simulation time: " << elapsed_seconds.count();
  std::cout << " N " << N << " steps " << nSteps << std::endl;
  // if we want to read the data and print the first/last element:
  //  ret += BPRead(fname, N, nSteps);
  std::remove((fname + "/data.0.0").c_str());
  return ret;
}
