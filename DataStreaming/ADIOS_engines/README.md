# Running the benchmarks

To compile all the codes and create executables for each engine:
```bash
mkdir build
cd build
cmake ..
make -j4
```

Running each engine.

**SST**
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
