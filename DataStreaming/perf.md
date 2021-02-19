# Performance runs

## Summit

Modules needed to be able to make runs with ADIOS
```bash
module load gcc
module avail adios2
   adios/1.13.1-py2    adios2/2.4.0    adios2/2.5.0    adios2/2.7.0 (D)
module load adios2/2.7.0
module load cmake
```

To compile the codes used to measure performance
```bash
cd performance
mkdir build
cd build
cmake ..
make -j4
```

## Laptop

```bash
cmake -D adios2_ROOT=~/work/adios/ADIOS2-init/install/lib/cmake ..
make -j4
```
