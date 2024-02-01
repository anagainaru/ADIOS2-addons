## ADIOS2 reorganize script using GPU

Changes to the code implemented in `source/utils/adios_reorganize/Reorganize.cpp` to read and write on the GPU.

This is useful to reproduce performance results when using GPU pointers for a given dataset.


### Code changes
```
source/utils/adios_reorganize/Reorganize.cpp:        varinfo[varidx].readbuf = calloc(1, varinfo[varidx].writesize);                            \
source/utils/adios_reorganize/Reorganize.cpp:    rStream.EndStep(); // read in data into allocated pointers
```
