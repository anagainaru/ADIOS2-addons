/*
 * Simple example of writing and reading data
 * through ADIOS2 BP engine with multiple simulations steps
 * for every IO step.
 */

#include "cudaRoutines.h"

#include <ios>
#include <iostream>
#include <algorithm>
#include <vector>

#include <adios2.h>

#include <cuda_runtime.h>

int BPWrite(const std::string fname, const size_t N, int nSteps)
{
    float *gpuSimData;
    cudaMalloc(&gpuSimData, N * sizeof(float));
    cudaMemset(gpuSimData, 0, N);
	std::vector<float> cpuSimData(N, 0);

    adios2::ADIOS adios;
    adios2::IO io = adios.DeclareIO("WriteIO");
    io.SetEngine("BP5");

    const adios2::Dims shape{static_cast<size_t>(N)};
    const adios2::Dims start{static_cast<size_t>(0)};
    const adios2::Dims count{N};
    auto data = io.DefineVariable<float>("data", shape, start, count);

    adios2::Engine bpWriter = io.Open(fname, adios2::Mode::Write);

    for (size_t step = 0; step < nSteps; ++step)
    {
        adios2::Box<adios2::Dims> sel({0}, {N});
        data.SetSelection(sel);

        bpWriter.BeginStep();
        if (step % 2 == 0)
        {
            bpWriter.Put(data, gpuSimData);
        } else
        {
            bpWriter.Put(data, cpuSimData.data());
        }
        bpWriter.EndStep();

		cuda_increment(N, 1, 0, gpuSimData, 10);
        std::transform(cpuSimData.begin(), cpuSimData.end(), cpuSimData.begin(), [](int i) -> int { return i + 1; });
    }

    bpWriter.Close();
    return 0;
}

int BPRead(const std::string fname, const size_t N, int nSteps)
{
    adios2::ADIOS adios;
    adios2::IO io = adios.DeclareIO("ReadIO");
    io.SetEngine("BP5");

    adios2::Engine bpReader = io.Open(fname, adios2::Mode::Read);

    unsigned int step = 0;
    float *gpuSimData;
    cudaMalloc(&gpuSimData, N * sizeof(float));
    cudaMemset(gpuSimData, 0, N);
    for (; bpReader.BeginStep() == adios2::StepStatus::OK; ++step)
    {
        auto data = io.InquireVariable<float>("data");
        std::vector<float> simData(N);
        const adios2::Dims start{0};
        const adios2::Dims count{N};
        const adios2::Box<adios2::Dims> sel(start, count);
        data.SetSelection(sel);

        //data.SetMemorySpace(adios2::MemorySpace::CUDA);
        if (step % 2 != 0)
			bpReader.Get(data, gpuSimData);
		else
			bpReader.Get(data, simData);
        bpReader.EndStep();
		
        if (step % 2 != 0)
        	cudaMemcpy(simData.data(), gpuSimData, N * sizeof(float),
            	       cudaMemcpyDeviceToHost);
        std::cout << "Simualation step " << step << " : ";
        std::cout << simData.size() << " elements: " << simData[1] << std::endl;
    }
    bpReader.Close();
    return 0;
}

int main(int argc, char **argv)
{
    const std::string fname("CudaBp5wr.bp");
    const int device_id = 1;
    cudaSetDevice(device_id);
    const size_t N = 6000;
    int nSteps = 10, ret = 0;

    ret += BPWrite(fname, N, nSteps);
    ret += BPRead(fname, N, nSteps);
    return ret;
}
