# Performance runs

## Summit

Modules needed to be able to make runs with ADIOS
```bash
module load gcc
module load adios2/2.7.0
module load cmake

module avail adios2
   adios/1.13.1-py2    adios2/2.4.0    adios2/2.5.0    adios2/2.7.0 (D)
```

To compile the codes used to measure performance
```bash
cd performance
mkdir build
cd build
cmake ..
make -j4
```

To run the code, due to incomptibility between libfabrics and mpi.
```
export LD_PRELOAD=/usr/lib64/libibverbs.so.1:/usr/lib64/librdmacm.so.1
```

## Laptop

```bash
cmake -D adios2_ROOT=~/work/adios/ADIOS2-init/install/lib/cmake ..
make -j4
```

# Engines

## SST

On the **Writer side**, each rank writes N floats of random values.
On the **Reader side**, each rank reads an equal portion of the written data. Parameters that can be changed are number of ranks for read/write and total array size.

Test caes include:
- One reader and one writer (ranks from SC19 paper) exchanging 10MB/rank or 1GB/rank data.
- 10 Writers and one reader reading from all 10 variables
- One writer and 10, 100, 1000 readers (keeping total ranks equal)
