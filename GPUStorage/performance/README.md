# GDS Performance

Measuring the performance of accessing a buffer stored on GPU space to the storage.

## Write performance

Measure the performance of writing an array of N floats residing on GPU memory space, written all at once.
1. GDS POSIX: Use Nvidia's GDS
2. CPUCopy POSIX: Copy the data from GPU to CPU memory space and use POSIX write
3. GDS ADIOS: Send the GPU buffer to ADIOS
4. CPU ADIOS: Send the CPU buffer to ADIOS

![Write performance](https://raw.githubusercontent.com/anagainaru/ADIOS2-addons/main/GPUStorage/docs/gds_write_perf.png)
