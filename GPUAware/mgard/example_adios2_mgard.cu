#include <cstdint>
#include <cstring>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <cuda_runtime.h>

#include <adios2.h>

std::vector<double> put_with_mgard(const std::string tolerance,
		const size_t Nx, const size_t Ny)
{
    const std::string fname("BPWRMGARD2D_" + tolerance + ".bp");

    std::vector<double> r64s(Nx * Ny);
    double *gpuSimData;
    cudaMalloc(&gpuSimData, Nx * Ny * sizeof(double));
    cudaMemset(gpuSimData, 0, Nx * Ny);

    // range 0 to 100*50
    std::iota(r64s.begin(), r64s.end(), 0.);
    cudaMemcpy(gpuSimData, r64s.data(), Nx * Ny * sizeof(double),
                   cudaMemcpyHostToDevice);

    adios2::ADIOS adios;
    adios2::IO io = adios.DeclareIO("WriteIO");

    const adios2::Dims shape{Nx, Ny};
    const adios2::Dims start{0, 0};
    const adios2::Dims count{Nx, Ny};

    auto var_r64 = io.DefineVariable<double>("r64", shape, start, count,
                                             adios2::ConstantDims);

    // add operations
    adios2::Operator mgardOp =
        adios.DefineOperator("mgardCompressor", adios2::ops::LossyMGARD);
    var_r64.AddOperation(mgardOp,
                         {{adios2::ops::mgard::key::tolerance, tolerance}});

    adios2::Engine bpWriter = io.Open(fname, adios2::Mode::Write);

	bpWriter.BeginStep();
	var_r64.SetMemorySpace(adios2::MemorySpace::CUDA);
	bpWriter.Put<double>("r64", gpuSimData);
	bpWriter.EndStep();

    bpWriter.Close();
	return r64s;
}

std::vector<double> get_with_mgard(const std::string tolerance, const size_t Nx, const size_t Ny)
{
    const std::string fname("BPWRMGARD2D_" + tolerance + ".bp");

    adios2::ADIOS adios;
	adios2::IO io = adios.DeclareIO("ReadIO");

	adios2::Engine bpReader = io.Open(fname, adios2::Mode::Read);

	std::vector<double> decompressedR64s;
	while (bpReader.BeginStep() == adios2::StepStatus::OK)
	{
		auto var_r64 = io.InquireVariable<double>("r64");

		const adios2::Dims start{0, 0};
		const adios2::Dims count{Nx, Ny};
		const adios2::Box<adios2::Dims> sel(start, count);
		var_r64.SetSelection(sel);

		double *gpuSimData = nullptr;
		cudaMalloc(&gpuSimData, Nx * Ny * sizeof(double));
		bpReader.Get(var_r64, gpuSimData);
		// bpReader.Get(var_r64, decompressedR64s);
		bpReader.EndStep();
		cudaMemcpy(decompressedR64s.data(), gpuSimData, Nx * Ny * sizeof(double),
		   cudaMemcpyDeviceToHost);
	}
	return decompressedR64s;
}

int main(int argc, char **argv)
{
    const size_t Nx = 100;
    const size_t Ny = 50;
	const std::string tolerance("0.001");
    auto r64s = put_with_mgard(tolerance, Nx, Ny);
    auto decompressedR64s = get_with_mgard(tolerance, Nx, Ny);

	double maxDiff;	
	for (size_t i = 0; i < Nx * Ny; ++i)
	{
		double diff = std::abs(r64s[i] - decompressedR64s[i]);

		if (diff > maxDiff)
		{
			maxDiff = diff;
		}
	}

	auto itMax = std::max_element(r64s.begin(), r64s.end());

	const double relativeMaxDiff = maxDiff / *itMax;
	std::cout << "Relative Max Diff " << relativeMaxDiff
			  << " tolerance " << tolerance << "\n";
    return 1;
}
