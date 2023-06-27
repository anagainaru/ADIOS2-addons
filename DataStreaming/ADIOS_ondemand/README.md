# Streaming on demand

Compiling

```
cmake -D adios2_ROOT=/path/to/adios2/install -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc ..
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
