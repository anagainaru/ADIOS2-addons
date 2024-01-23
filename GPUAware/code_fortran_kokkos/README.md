## Fortran example with MPI/ADIOS2/Kokkos

### Build and run

The path to the adios2, Kokkos and flcl installation needs to be provided to cmake (`adios2_ROOT`, `Kokkos_ROOT`, `flcl_ROOT`).

Example run on an Nvidia architecure:

```bash
$ mpirun -np 2 ./adios2_writeread_f
 Rank            0  Writing on Cuda memory space (one variable) and on Host space (one variable)
 Rank            1  Writing on Cuda memory space (one variable) and on Host space (one variable)
 Reading on Default execution space
 Reading data written on Host
1 1 1 1
2 2 2 2
3 3 3 3
2 2 2 2
3 3 3 3
4 4 4 4
 Reading data written on the Cuda memory space
0 0 0 0
1 1 1 1
2 2 2 2
1 1 1 1
2 2 2 2
3 3 3 3
$ ~/adios/ADIOS2/install-kokkos/bin/bpls BPFortranKokkos.bp/ -d -n 6
  int32_t  bpFloats     {4, 6}
Detect
is host: 1
    (0,0)    1 2 3 2 3 4
    (1,0)    1 2 3 2 3 4
    (2,0)    1 2 3 2 3 4
    (3,0)    1 2 3 2 3 4

  int32_t  bpFloatsGPU  {4, 6}
Detect
is host: 1
    (0,0)    0 1 2 1 2 3
    (1,0)    0 1 2 1 2 3
    (2,0)    0 1 2 1 2 3
    (3,0)    0 1 2 1 2 3
```

### Kokkos configuration

In order to print the Kokkos configuration used to run the example, add the line to the main fortran file:
`call kokkos_print_configuration('flcl-config-', 'kokkos.out')`

```bash
$ cat flcl-config-kokkos.out
  Kokkos Version: 4.2.99
Compiler:
  KOKKOS_COMPILER_GNU: 940
  KOKKOS_COMPILER_NVCC: 1170
Architecture:
  CPU architecture: none
  Default Device: N6Kokkos4CudaE
  GPU architecture: TURING75
  platform: 64bit
Atomics:
Vectorization:
  KOKKOS_ENABLE_PRAGMA_IVDEP: no
  KOKKOS_ENABLE_PRAGMA_LOOPCOUNT: no
  KOKKOS_ENABLE_PRAGMA_UNROLL: no
  KOKKOS_ENABLE_PRAGMA_VECTOR: no
Memory:
  KOKKOS_ENABLE_HBWSPACE: no
  KOKKOS_ENABLE_INTEL_MM_ALLOC: no
Options:
  KOKKOS_ENABLE_ASM: yes
  KOKKOS_ENABLE_CXX17: yes
  KOKKOS_ENABLE_CXX20: no
  KOKKOS_ENABLE_CXX23: no
  KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK: no
  KOKKOS_ENABLE_HWLOC: no
  KOKKOS_ENABLE_LIBDL: yes
  KOKKOS_ENABLE_LIBRT: no
Host Serial Execution Space:
  KOKKOS_ENABLE_SERIAL: yes

Serial Runtime Configuration:
Device Execution Space:
  KOKKOS_ENABLE_CUDA: yes
Cuda Options:
  KOKKOS_ENABLE_CUDA_LAMBDA: yes
  KOKKOS_ENABLE_CUDA_LDG_INTRINSIC: yes
  KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE: no
  KOKKOS_ENABLE_CUDA_UVM: no
  KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA: yes
  KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC: yes

Cuda Runtime Configuration:
macro  KOKKOS_ENABLE_CUDA      : defined
macro  CUDA_VERSION          = 11070 = version 11.7
Kokkos::Cuda[ 0 ] NVIDIA GeForce RTX 2080 Ti capability 7.5, Total Global Memory: 10.76 G, Shared Memory per Block: 48 K : Selected
Kokkos::Cuda[ 1 ] NVIDIA GeForce RTX 2080 Ti capability 7.5, Total Global Memory: 10.74 G, Shared Memory per Block: 48 K
Kokkos::Cuda[ 2 ] Quadro RTX 5000 capability 7.5, Total Global Memory: 15.75 G, Shared Memory per Block: 48 K

```
