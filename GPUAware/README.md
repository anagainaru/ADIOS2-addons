# GPU-aware ADIOS

**1. GPU-aware ADIOS** <br/>
Capable of receiving GPU buffers during Put calls. Underneath, ADIOS copies the content from the user provided GPU buffer to an internal ADIOS buffer. Metadata (min/max) is done using GPU kernels directly on the user buffer.

<img width="721" alt="GPU aware ADIOS" src="https://user-images.githubusercontent.com/16229479/138385188-5ce0c1c6-59be-4709-932a-6122ef5dd7e5.png">

In folder `ADIOS.cuda`

**2. GDS-capable ADIOS** <br/>
Same as 1, ADIOS can receive GPU buffers in Put or Get calls. Underneath, it uses a new transport to call Nvidia's GDS calls to move the data directly to NVME. Metadata (min/max) needs to be done on the GPU useing the user buffer.

<img width="713" alt="GDS ADIOS" src="https://user-images.githubusercontent.com/16229479/138386014-93fe57fc-cd85-48ea-be68-bf25d8f4322a.png">

In folder `ADIOS.GDS`

There are two options where the data is written with GDS:
- In a separate per rank GPU file {name}.bp/data.gpu.0
- In the same data file used by the bp engine, writing at a certain offset

## Application impact

Performance results for applications using GPU buffers: currently the OpenPMD benchmark used by WarpX.


**Others folders in this path**
- `docs` figures used by markdown files.
- `performance` I/O kernels used to measure the performance of each GPU level
