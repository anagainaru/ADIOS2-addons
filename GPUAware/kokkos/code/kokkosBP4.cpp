/*
 * Simple example of writing and reading data
 * through ADIOS2 BP engine with multiple simulations steps
 * for every IO step.
 */

#include <ios>
#include <iostream>
#include <vector>

#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>
#include <Kokkos_Core.hpp>

int BPWrite(const std::string fname, const size_t N, int nSteps)
{
    // Initialize the simulation data
    Kokkos::View<float*, Kokkos::CudaSpace> gpuSimData("simBuffer", N);

    // Set up the ADIOS structures
    adios2::ADIOS adios;
    adios2::IO io = adios.DeclareIO("WriteIO");
    io.SetEngine("BPFile");

    // Declare an array for the ADIOS data of size (NumOfProcesses * N)
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
        bpWriter.BeginStep();
        bpWriter.Put(data, gpuSimData);
        bpWriter.EndStep();

        // Update values in the simulation data
        Kokkos::parallel_for("updateBuffer",
			Kokkos::RangePolicy<Kokkos::Cuda>(0,N),
			KOKKOS_LAMBDA(int i){
				gpuSimData(i) += 5;
		});
    }

    bpWriter.Close();
    return 0;
}

int BPRead(const std::string fname, const size_t N, int nSteps)
{
    // Create ADIOS structures
    adios2::ADIOS adios;
    adios2::IO io = adios.DeclareIO("ReadIO");
    io.SetEngine("BPFile");

    adios2::Engine bpReader = io.Open(fname, adios2::Mode::Read);
    auto data = io.InquireVariable<float>("data");
    std::cout << "Steps expected by the reader: " << bpReader.Steps()
              << std::endl;
    std::cout << "Expecting data per step: " << data.Shape()[0];
    std::cout << " elements" << std::endl;

    int write_step = bpReader.Steps();
    // Create the local buffer and initialize the access point in the ADIOS file
    const adios2::Dims start{0};
    const adios2::Dims count{N};
    const adios2::Box<adios2::Dims> sel(start, count);
    data.SetSelection(sel);

    // Initialize the simulation data
    Kokkos::View<float*, Kokkos::CudaSpace> gpuSimData("simBuffer", N);

    // Read the data in each of the ADIOS steps
    for (size_t step = 0; step < write_step; step++)
    {
        data.SetStepSelection({step, 1});
        bpReader.Get(data, gpuSimData); //, adios2::Mode::Deferred);
        bpReader.PerformGets();
        auto simData = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{}, gpuSimData);
        std::cout << "Simualation step " << step << " : ";
        std::cout << simData.size() << " elements: " << simData[1] << std::endl;
    }

    bpReader.Close();
    return 0;
}

int main(int argc, char **argv)
{
    const std::string fname("CudaBp4wr.bp");
    const size_t N = 6000;
    int nSteps = 10, ret = 0;
    Kokkos::initialize( argc, argv );
    {
        ret += BPWrite(fname, N, nSteps);
        ret += BPRead(fname, N, nSteps);
    }
    Kokkos::finalize();
    return ret;
}
