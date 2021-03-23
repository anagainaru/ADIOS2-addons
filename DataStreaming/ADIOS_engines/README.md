# Running the codes

## Compie the codes

### Summit

To compile all the codes and create executables for each engine, load all the necessary packages:
```bash
module load cmake
module load gcc
module load adios2/2.7.0
```

Build the codes:
```bash
mkdir build
cd build
cmake ..
make -j4
```

To run the code, due to incomptibility between libfabrics and mpi.
```
export LD_PRELOAD=/usr/lib64/libibverbs.so.1:/usr/lib64/librdmacm.so.1
```

The codes have been tested with the following versions:
```
cmake version 3.18.2
gcc (GCC) 6.4.0
```

### Personal system
```bash
cmake -D adios2_ROOT=~/work/adios/ADIOS2-init/install/lib/cmake ..
make -j4
```

## Running each engine

**SST**

On the **Writer side**, each rank writes N floats of random values.
On the **Reader side**, each rank reads an equal portion of the written data. Parameters that can be changed are number of ranks for read/write and total array size.

Test caes include:
- One reader and one writer (ranks from SC19 paper) exchanging 10MB/rank or 1GB/rank data.
- 10 Writers and one reader reading from all 10 variables
- One writer and 10, 100, 1000 readers (keeping total ranks equal)

```bash
$ ./sstWriter & ./sstReader
Incoming variable is of size 10
Reader rank 0 reading 10 floats starting at element 0

$ mpirun -np 2 ./sstWriter & mpirun -np 2 ./sstReader
Incoming variable is of size 20
Reader rank 0 reading 10 floats starting at element 0
Incoming variable is of size 20
Reader rank 1 reading 10 floats starting at element 10
```

**SSC**
```bash
$ mpirun -np 2 sscReadWriter
Incoming variable is of size 10
Reader rank 0 reading 10 floats starting at element 0: first element 10
```

**Inline**
```bash
$ ./iReadWriter
Data StepsStart 0 from rank 0: 9 1 2 3 4 5 6 7 8 9
inlineString: Hello from rank: 0 and timestep: 0
Data StepsStart 0 from rank 0: 10 1 2 3 4 5 6 7 8 9
inlineString: Hello from rank: 0 and timestep: 1
Data StepsStart 0 from rank 0: 11 1 2 3 4 5 6 7 8 9
inlineString: Hello from rank: 0 and timestep: 2

$ mpirun -np 2 ./iReadWriter
Data StepsStart 0 from rank 0: 9 1 2 3 4 5 6 7 8 9
inlineString: Hello from rank: 0 and timestep: 0
Data StepsStart 0 from rank 1: 0 10 2 3 4 5 6 7 8 9
Data StepsStart 0 from rank 0: 10 1 2 3 4 5 6 7 8 9
inlineString: Hello from rank: 0 and timestep: 1
Data StepsStart 0 from rank 1: 0 11 2 3 4 5 6 7 8 9
Data StepsStart 0 from rank 1: 0 12 2 3 4 5 6 7 8 9
Data StepsStart 0 from rank 0: 11 1 2 3 4 5 6 7 8 9
inlineString: Hello from rank: 0 and timestep: 2
```

**DataMan**
```bash
./dmWriter & ./dmReader
libc++abi.dylib: terminating with uncaught exception of type std::invalid_argument: ERROR: this version didn't compile with DataMan library, can't use DataMan engine
```

**DataSpaces**
```bash
./dsWriter & ./dsReader
Invalid argument exception, STOPPING PROGRAM from rank 0
ERROR: this version didn't compile with DataSpaces library, can't use DataSpaces engine
```
