# GPU-aware ADIOS with Kokkos

## Install Kokkos

```
cmake -B build -DCMAKE_INSTALL_PREFIX=${PWD}/install -DKokkos_ENABLE_THREADS=ON -DCMAKE_BUILD_TYPE=Release -D Kokkos_ENABLE_HWLOC=ON

cmake --build build --parallel 6
cmake --install build
```

## Design ADIOS with Kokkos

Two scenarios:
1. The application is giving Kokkos View to the Get/Put ADIOS functions instead of GPU buffers
2. ADIOS is build with Kokkos enabled

**Assumptions**
Multiple backends can be enabled during compilation but only one will be used during execution (i.e. pure CUDA code).

## 1. Kokkos View buffers

The simple solution overloads the Put/Get functions to work with `Kokkos::View` and simply
extracts the buffer pointer and memory space from it. It then calls the existing Put or Get function.

<img width="705" alt="Screen Shot 2022-04-08 at 1 23 59 PM" src="https://user-images.githubusercontent.com/16229479/162491439-3240d802-8d8f-42fa-a8f1-682fe3558994.png">

If we want to use ADIOS with Kokkos for buffer management, we will store a `Kokkos::View` inside the ADIOS variable that will store all the information needed to deal with GPU buffers.

**Edit** Eventually the `Kokkos::View` can be used for both CPU and GPU buffers

## 2. ADIOS build with Kokkos

<img width="580" alt="Screen Shot 2022-04-08 at 2 11 55 PM" src="https://user-images.githubusercontent.com/16229479/162498266-9ce8e7a1-ad9c-43d5-afc2-39510252bbaf.png">

If ADIOS is not build with Kokkos, the GPU buffer will be handled by the corresponding functions (for payload and matadata handling) based on the memory space.

If ADIOS is build with Kokkos, the buffer and memory space provided by the user will create a `Kokkos::View` with the pointer and the specified memory space that will be stored in the variable.

```c++
#ifdef ADIOS2_HAVE_CUDA
if ((var.m_MemorySpace == MemorySpace::CUDA) || (var.IsCUDAPointer(data)))
   Kokkos::View<T *, Kokkos::CudaSpace> gpuData(data);
#elif ADIOS2_HAVE_SYCL
if ((var.m_MemorySpace == MemorySpace::Experimental::Sycl) || (var.IsSyclPointer(data)))
   Kokkos::View<T *, Kokkos::CudaSpace> gpuData(data);
#endif
```
