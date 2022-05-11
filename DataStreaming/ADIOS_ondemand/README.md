# Streaming on demand

Using a configuration file 

Compiling

```
cmake -D adios2_ROOT=/Users/95j/work/adios/ADIOS2-main/install -D MPI_ROOT=/Users/95j/opt/usr/local -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc ..
make -j4
```

Results when using one or two consumers
```
$ ./sstWriter 100 2 & ./sstReader 100 2
SST,Write,1,100,2,100,6976,56482
SST,Read,1,100,2,98,2026,57045

$ ./sstWriter 100 2 & ./sstReader 100 2 & ./sstReader 100 2
SST,Write,1,100,2,100,6551,56033
SST,Read,1,100,2,2,177,57931
SST,Read,1,100,2,97,2008,56521
```

For some reason not all steps are read. In the previous example the number of steps writen/read by the producer/consumer are the 6th element in the CSV data (100/2/97). Each process is plotting the steps it puts/gets for one variable per step and 6 steps:

```
./sstWriter 100 1 & ./sstReader 100 1 p1 & ./sstReader 100 1 p2
p0: Put step 0
p0: Put step 1
p0: Put step 2
p1: Get step 0
p0: Put step 3
p1: Get step 2
p2: Get step 1
p0: Put step 4
p2: Get step 2
p0: Put step 5
SST,Write,1,100,1,6,731,7499
SST,Read,1,100,1,2,135,9184
p2: Get step 5
SST,Read,1,100,1,3,286,8430
```
Step 3 is not read by any of the consumers in the previous example.
