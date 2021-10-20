# GPU-aware ADIOS

1. GPU-aware ADIOS.
Capable of receiving GPU buffers during Put calls. Underneath, ADIOS copies the content from the user provided GPU buffer to an internal ADIOS buffer. Metadata (min/max) is done using GPU kernels directly on the user buffer.

2. GDS-capable ADIOS.
Same as 1, ADIOS can receive GPU buffers in Put or Get calls. Underneath, it uses a new transport to call Nvidia's GDS calls to move the data directly to NVME. Metadata (min/max) needs to be done on the GPU useing the user buffer.

There are two options where the data is written with GDS:
- In a separate per rank GPU file {name}.bp/data.gpu.0
- In the same data file used by the bp engine, writing at a certain offset


