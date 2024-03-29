# GPU-aware ADIOS with Kokkos

There are 3 options for implementations: 
1) include `Kokkos::View` in ADIOS2 CXX bindings. (described in [kokkos-aware-adios.md](https://github.com/anagainaru/ADIOS2-addons/blob/kokkos-view/GPUAware/kokkos/kokkos-aware-adios.md))
2) Allowing Get/Put to receive any container type of object and overwrite the logic for `Kokkos::View` in a separate file. (described in [container-type.md](https://github.com/anagainaru/ADIOS2-addons/blob/kokkos-view/GPUAware/kokkos/container-type.md)) 
3) Allowing Get/Put to receive an `ADIOSView` stub defined in a kokkos header that needs to be included by the user. (defined in [adios-view.md](https://github.com/anagainaru/ADIOS2-addons/blob/kokkos-view/GPUAware/kokkos/adios-view.md))

## Install ADIOS with Kokkos

Install Kokkos (only threads backend allowed)
```
cmake -B build -DCMAKE_INSTALL_PREFIX=${PWD}/install -DKokkos_ENABLE_THREADS=ON -DCMAKE_BUILD_TYPE=Release -D Kokkos_ENABLE_HWLOC=ON -D CMAKE_CXX_STANDARD=17 -D CMAKE_CXX_EXTENSIONS=OFF -D CMAKE_POSITION_INDEPENDENT_CODE=TRUE -D BUILD_SHARED_LIBS=ON

cmake --build build --parallel 6
cmake --install build
```

Install ADIOS2 with Kokkos support
```bash
cmake -D Kokkos_ROOT=/path/to/kokkos/install -D ADIOS2_USE_Kokkos=ON -D CMAKE_POSITION_INDEPENDENT_CODE=TRUE -D BUILD_SHARED_LIBS=ON -D CMAKE_INSTALL_PREFIX=${PWD}/../install ..
make -j4
```

**Install Kokkos on Summit (cuda enabled)**

The `configure_kokkos.sh` script can be used to install Kokkos on Summit.

Install ADIOS, needs the `CXX_STANDARD` to 17 and the GCC compiler to point to the kokkos nvcc wrapper. 
```
cmake -D CMAKE_CXX_STANDARD=17 -D CMAKE_CXX_EXTENSIONS=OFF  -DCMAKE_C_COMPILER=gcc -DADIOS2_USE_CUDA=OFF -DADIOS2_BUILD_EXAMPLES=OFF 
-DBUILD_TESTING=OFF -DADIOS2_USE_SST=OFF -DKokkos_ROOT=/path/to/kokkos/install/ -D CMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CXX_COMPILER=/path/to/kokkos/bin/nvcc_wrapper ..
make -j4
cmake -D CMAKE_INSTALL_PREFIX=${ADIOS_HOME}/install
make install
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
