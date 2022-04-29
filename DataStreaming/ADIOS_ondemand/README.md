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

For some reason not all steps are read. The consumers are reading more steps than allowed. Need to investigate.
