# Measure the performance of the GPU-backend

The read and write files are separated and can be run consecutively for BP5 or concurrent for SST and Dataman.

**The write functions** take the following parameters:
- engine (BP5/SST/DataMan)
- array_size (either one integer for the 1D case or two for the 2D case)
- number_steps
- memory_space (host/device)
- output_file (optional otherwise default name is used)

Example run (laptop run, ignore the performance):
```
$ mpirun -np 2 ./adios2_writerKokkos1D BP5 100000 2 host test.bp
Engine: BP5
Memory space: HostSpace
Write1D BP5 Serial 0.38147 0.00760176 0.0490056 units:MB:s:GB/s
Write1D BP5 Serial 0.38147 0.00830747 0.0448426 units:MB:s:GB/s
```

**The read functions** take the following parameters:
- engine (BP5/SST/DataMan)
- input_file
- array_size (either one integer for the 1D case or two for the 2D case)
- memory_space (host/device)
  
Example run:
```
$ mpirun -np 2 ./adios2_readerKokkos2D BP5 test.bp 10 10000 host
Using engine BP5
Memory space: HostSpace
ReadBP5 Serial 0.38147 0.000370844 1.00454 units:MB:s:GB/s
ReadBP5 Serial 0.38147 0.000364622 1.02169 units:MB:s:GB/s
```

## Frontier

Module loaded to compile the codes. If Dataman is not needed ZeroMQ does not need to be loaded.
```
module load libzmq/4.3.4
module load PrgEnv-cray
module load cmake/3.23.2
module unload darshan-runtime/3.4.0
module load ums ums002
module load tau
```

ADIOS2 and Kokkos need to be installed with the same modules.
```
cmake -Dadios2_ROOT=/lustre/orion/csc303/proj-shared/againaru/ADIOS2/install-kokkos-frontier -DKokkos_ROOT=/lustre/orion/csc303/proj-shared/againaru/ADIOS2/install-kokkos-frontier -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc ..
```

Example run:
```
$ srun -n1 --gpus 1 tau_exec -T mpi -io ./adios2_writerKokkos1D BP5 1000000 2 device device-6-1d.bp
Write1D BP5 HIP 3814.7 2.83025 units:MB:s 1.31624 GB/s
Write1D BP5 HIP 3814.7 2.6827 units:MB:s 1.38864 GB/s
$ pprof
Reading Profile files in profile.*

NODE 0;CONTEXT 0;THREAD 0:
---------------------------------------------------------------------------------------
%Time    Exclusive    Inclusive       #Call      #Subrs  Inclusive Name
              msec   total msec                          usec/call
---------------------------------------------------------------------------------------
100.0        0.919        7,354           1           1    7354712 .TAU application
100.0        1,910        7,353           1          36    7353793 int taupreload_main(int, char **, char **)
 58.2        4,277        4,278           2          48    2139074 BP5Writer::EndStep
 12.5        0.216          920           1           3     920656 IO::Open
 12.5            1          920           1          48     920418 BP5Writer::Open
 12.5          918          918           4           0     229513 pthread_join
  1.6          117          117           2           0      58521 Kokkos::parallel_reduce N6Kokkos4Impl31CombinedReductionFunctorWrapperIZN12_GLOBAL__N_116KokkosMinMaxImplIfEEvPKT_mRS4_S7_EUliRfS8_E_NS_9HostSpaceEJNS_3MaxIfSA_EENS_3MinIfSA_EEEEE [type = HIP, device = 0]
  1.4          104          104           1           0     104906 Kokkos::parallel_for initBuffer [type = HIP, device = 0]
  0.2           14           14           2           0       7215 Kokkos::parallel_for updateBuffer [type = HIP, device = 0]
  0.0            3            3           1           0       3056 Kokkos::parallel_for Kokkos::View::initialization [simBuffer] via memset [type = HIP, device = 0]
  0.0            2            2           1           9       2432 BP5Writer::Close
  0.0            1            1           1           0       1476 MPI_Init_thread()
  0.0        0.519        0.519          47           0         11 MPI_Comm_size()
  0.0        0.462        0.462          41           0         11 MPI_Comm_rank()
  0.0        0.149        0.197           4           4         49 MPI_Gather()
  0.0        0.133        0.177           4           4         44 MPI_Bcast()
  0.0        0.173        0.173          13           0         13 MPI Collective Sync
  0.0        0.154        0.154           1           0        154 MPI_Finalize()
  0.0          0.1        0.145           2           2         72 MPI_Comm_dup()
  0.0        0.091        0.116           2           2         58 MPI_Reduce()
  0.0         0.11         0.11           8           0         14 MPI_Comm_free()
  0.0        0.094        0.094           8           0         12 MPI_Finalized()
  0.0        0.082        0.082           5           0         16 MPI_Comm_split()
  0.0        0.042        0.053           1           1         53 MPI_Gatherv()
  0.0        0.044        0.044           1           0         44 IO::DefineVariable
  0.0        0.026        0.026           2           0         13 void adios2::core::engine::BP5Writer::MarshalAttributes()
  0.0        0.019        0.019           1           0         19 MPI_Info_create()
  0.0        0.015        0.015           1           0         15 MPI_Barrier()
---------------------------------------------------------------------------------------

USER EVENTS Profile :NODE 0, CONTEXT 0, THREAD 0
---------------------------------------------------------------------------------------
NumSamples   MaxValue   MinValue  MeanValue  Std. Dev.  Event Name
---------------------------------------------------------------------------------------
         4          8          4          7      1.732  Message size for broadcast
         5        698          8        146        276  Message size for gather
         2          8          8          8          0  Message size for reduce
---------------------------------------------------------------------------------------
```
