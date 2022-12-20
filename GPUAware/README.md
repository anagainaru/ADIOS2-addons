# GPU-aware ADIOS

Currently only CUDA supported (through raw pointers or through Kokkos View with CUDA backend).

**Build ADIOS2 with CUDA support**

Specify flags to use CUDA and to set the CUDA architecture to 70 (for NVIDIA Volta V100).

```
module load cuda/11.0.3 gcc/9.1.0 cmake/3.23.2 
cmake -D CMAKE_CUDA_ARCHITECTURES=70 -D ADIOS2_USE_CUDA=ON -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
make -j4
```

**Build ADIOS2 with Kokkos support**

Install Kokkos and point to the Kokkos folder when building ADIOS2

```
module load gcc/9.1.0 cmake/3.23.2 cuda/11.0.3
cmake -DKokkos_ROOT=/path/to/kokkos/install ../ADIOS2/
make -j4
```

## API

Same API as for CPU buffers.

```c++
#include <adios2.h>
#include <cuda_runtime.h>

...
    float *gpuSimData;
    cudaMalloc(&gpuSimData, N * sizeof(float));
    // cudamemcpy to fill the simulation data with initial values
    auto data = io.DefineVariable<float>("data", shape, start, count);
    for (steps){
        bpWriter.BeginStep();
        bpWriter.Put(data, gpuSimData);
        bpWriter.EndStep();
        // call CUDA kernels to update the gpu array
    }
```

To build the codes, cmake will require linking CUDA or Kokkos. Examples are in the code example folders.

The path to ADIOS2 installation needs to be specified during build:
```
module load cuda/11.0.2 gcc/9.1.0 cmake/3.23.2
cmake -Dadios2_ROOT=/ccs/home/againaru/adios/ADIOS2/install -D CMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..

```

For Kokkos examples other paths might need to be included (example, path to Kokkos install).

**Code examples**

For CUDA the `code_example_cuda` uses BP5 to write data from a GPU and CPU buffer alternatively into the same ADIOS2 variable.

```
$ cmake -Dadios2_ROOT=/ccs/home/againaru/adios/ADIOS2/install -D CMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
$ ./adiosCUDAex
Simualation step 0 : 6000 elements: 0
Simualation step 1 : 6000 elements: 1
Simualation step 2 : 6000 elements: 20
Simualation step 3 : 6000 elements: 3
Simualation step 4 : 6000 elements: 40
Simualation step 5 : 6000 elements: 5
Simualation step 6 : 6000 elements: 60
Simualation step 7 : 6000 elements: 7
Simualation step 8 : 6000 elements: 80
Simualation step 9 : 6000 elements: 9
```

For Kokkos, the `performance` folder has an example.

```
$ cmake -Dadios2_ROOT=/ccs/home/againaru/adios/ADIOS2/install -D CMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -D Kokkos_ROOT=path/to/kokkos ..
```

## Implementation details

**1. GPU-aware ADIOS** <br/>
Capable of receiving GPU buffers during Put calls. Underneath, ADIOS copies the content from the user provided GPU buffer to an internal ADIOS buffer. Metadata (min/max) is done using GPU kernels directly on the user buffer.

<img width="721" alt="GPU aware ADIOS" src="https://user-images.githubusercontent.com/16229479/138385188-5ce0c1c6-59be-4709-932a-6122ef5dd7e5.png">

In folder `ADIOS.cuda`

**2. GDS-capable ADIOS** <br/>
Same as 1, ADIOS can receive GPU buffers in Put or Get calls. Underneath, it uses a new transport to call Nvidia's GDS calls to move the data directly to NVME. Metadata (min/max) needs to be done on the GPU useing the user buffer.

<img width="713" alt="GDS ADIOS" src="https://user-images.githubusercontent.com/16229479/138386014-93fe57fc-cd85-48ea-be68-bf25d8f4322a.png">

In folder `ADIOS.GDS`

There are two options where the data is written with GDS:
- In a separate per rank GPU file {name}.bp/data.gpu.0
- In the same data file used by the bp engine, writing at a certain offset

## Application impact

Performance results for applications using GPU buffers: currently the OpenPMD benchmark used by WarpX.


**Others folders in this path**
- `docs` figures used by markdown files.
- `performance` I/O kernels used to measure the performance of each GPU level
