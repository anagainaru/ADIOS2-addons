/*
 * Simple example of writing and reading data
 * through ADIOS2 BP engine with multiple simulations steps
 * for every IO step.
 */

#include <ios>
#include <vector>
#include <iostream>

#include <adios2.h>
#include <Kokkos_Core.hpp>

int BPWrite(const std::string fname, const size_t N,
            int nSteps, float startVal){
  int write_step = 10; // write every 10th simulation step
  int cpu_step = 2; // write from cpu every other write step
  // Initialize the simulation data
  std::vector<float> simData(N, startVal);
  Kokkos::View<float*> gpuSimData("gpu_buffer", N);
 
  // Set up the ADIOS structures
  adios2::ADIOS adios;
  adios2::IO io = adios.DeclareIO("WriteIO");

  const adios2::Dims shape{static_cast<size_t>(N)};
  const adios2::Dims start{static_cast<size_t>(0)};
  const adios2::Dims count{N};
  auto data = io.DefineVariable<float>("data", shape, start, count);

  adios2::Engine bpWriter = io.Open(fname, adios2::Mode::Write);
  
  // Simulation steps
  for (size_t step = 0; step < nSteps; ++step)
  {
      // Make a 1D selection to describe the local dimensions of the
      // variable we write and its offsets in the global spaces
      adios2::Box<adios2::Dims> sel({0}, {N});
      data.SetSelection(sel);

      // Start IO step every write step
      if (step % write_step == 0){
        bpWriter.BeginStep();
        if (step % (write_step * cpu_step) == 0)
	  bpWriter.Put(data, simData.data(), adios2::Mode::Sync);
	else
	  bpWriter.Put(data, gpuSimData.data(), adios2::Mode::Sync);
        bpWriter.EndStep();
      }

      // Compute new values for the data
      for (int i = 0; i < N; i++)
        simData[i] += 1;

      Kokkos::parallel_for("update_buffer", N, KOKKOS_LAMBDA(int i){
	gpuSimData(i) += 5;
      });
  }

  bpWriter.Close();
  return 0;
}

int BPRead(const std::string fname, const size_t N, int nSteps){
  // Create ADIOS structures
  adios2::ADIOS adios;
  adios2::IO io = adios.DeclareIO("ReadIO");

  adios2::Engine bpReader = io.Open(fname, adios2::Mode::Read);

  auto data = io.InquireVariable<float>("data");
  std::cout << "Steps expected by the reader: " << bpReader.Steps() << std::endl;
  std::cout << "Expecting data per step: " << data.Shape()[0];
  std::cout  << " elements" << std::endl;

  int write_step = bpReader.Steps();
  // Create the local buffer and initialize the access point in the ADIOS file
  std::vector<float> simData(N); //set size to N
  const adios2::Dims start{0};
  const adios2::Dims count{N};
  const adios2::Box<adios2::Dims> sel(start, count);
  data.SetSelection(sel);
  
  // Read the data in each of the ADIOS steps
  for (size_t step = 0; step < write_step; step++)
  {
      data.SetStepSelection({step, 1});
      bpReader.Get(data, simData.data());
      bpReader.PerformGets();
      std::cout << "Simualation step " << step << " : ";
      std::cout << simData.size() << " elements: " << simData[1] << std::endl;
  }
  bpReader.Close();
  return 0;
}

int main(int argc, char **argv){
  Kokkos::ScopeGuard gds(argc, argv);

  const std::string fname("BPAnaWriteRead.bp");
  const size_t N = 100;
  int nSteps = 100, ret = 0;

  ret = BPWrite(fname, N, nSteps, 5);
  ret += BPRead(fname, N, nSteps);
  return ret;
}
