# GPU - Storage direct access

## Steps to run Nvidia GDS code

Documentation about the NVidia cuFile API can be found [here](https://docs.nvidia.com/gpudirect-storage/index.html).

Instructions to build Nvidia cuFile code are for sdg-tm76 at ORNL. Configuring and installing the Nvidia driver on sdg-tm76 can be found [here](https://docs.google.com/document/d/1j___qra3mpecBxy_J9MKi38wKoQafC4J3oKcDyhcEMw). The same intructions in markdown format can be found bellow.

Several code exampes using the cuFile API without or without multi-threading can be found on sdg-tm76 in the `/usr/local/cuda-11.1/gds/samples` folder.

Currently nvcc is not in `PATH` so it need to be added before compiling any GPU code:
```
export PATH=$PATH:/usr/local/cuda-11.1/bin
```

All the examples that use ADIOS with CUDA enabled require modifications to the ADIOS library (described [here](ADIOS.md)). The ADIOS library with CUDA enabled needs to be build and installed prior to running the examples. The example that uses kokkos with ADIOS calls needs the Kokkos library build and installed on the system. All examples have been tested on System76@ORNL with ADIOS installed in `/home/gainaru/adios/install/lib/cmake` and Kokkos in `/home/gainaru/kokkos/install/lib/cmake`. Change these paths to point to your own installations.


### Examples in this repo

**Read/Write cuda code**

Code in the `cuda_readwrite` folder.
Simple example that reads a file from NVME directly into GPU memory space then writes the same data directly from the GPU back to NVME in a different file.

Flags needed for compile:
```
g++ -Wall -I /usr/local/cuda/include/  -I /usr/local/cuda-11.1/.targets/x86_64-linux/lib/ hello_cuda.cc -o hello_cuda -L /usr/local/cuda-11.1/targets/x86_64-linux/lib/ -lcufile -L /usr/local/cuda/lib64/ -lcuda -L /usr/local/cuda/lib64/ -lcudart_static -lrt -lpthread -ldl

$ export PATH=$PATH:/usr/local/cuda-11.1/bin
$ make
$ sudo ./hello_cuda
Allocating and reading memory of size :32505856 gpu id: 3
Writing memory of size :32505856 gpu id: 3
```

**Read/Write kokkos code**

Code in the `kokkos_readwrite` folder. Same example as the cuda but using Kokkos for memory allocation on the GPU. This example write the data using both the GDS technology directly from the GPU memory space as well as with the CPU after using Kokkos to transfer the data.

Running the code
```
$ export PATH=$PATH:/usr/local/cuda-11.1/bin
$ make KOKKOS_DEVICES=Cuda -j8 KOKKOS_ARCH=Turing75

$ sudo ./hello.cuda
GPU direct read memory of size :32505856 gpu id: 3
GPU direct write memory of size :32505856 gpu id: 3
CPU Writing memory of size :32505856
21a3e2187bc12f803690bf809775694d81e06150dfc92c6f77322ac349890f4
21a3e2187bc12f803690bf809775694d81e06150dfc92c6f77322ac349890f4
21a3e2187bc12f803690bf809775694d81e06150dfc92c6f77322ac349890f4
SHA SUM Match
```

**Adios Cuda code**

Example of cuda code that uses ADIOS for read / write data to storage has two variants. The folder `cuda_adios` contains a stand alone example of using adios within a CUDA program and there is also the same example that is included inside the ADIOS2 examples folder. 

Both examples use the same code and they write to the storage every 10 simulation steps, alternating between the CPU and GPU. Running the stand alone code
```
export PATH=$PATH:/usr/local/cuda-11.1/bin
mkdir build
cd build
cmake -D adios2_ROOT=/home/gainaru/adios/install/lib/cmake ../
make -j
$ ./adios.cuda
Steps expected by the reader: 10
Expecting data per step: 100 elements
Simualation step 0 : 100 elements: 5
Simualation step 1 : 100 elements: 0
Simualation step 2 : 100 elements: 25
Simualation step 3 : 100 elements: 0
Simualation step 4 : 100 elements: 45
Simualation step 5 : 100 elements: 0
Simualation step 6 : 100 elements: 65
Simualation step 7 : 100 elements: 0
Simualation step 8 : 100 elements: 85
Simualation step 9 : 100 elements: 0
```

Running the code included in the ADIOS example. Output is the same for both.
```
git clone git@github.com:anagainaru/ADIOS2.git .
git checkout gpu
mkdir build
cd build
cmake ../ADIOS2
make -j
./bin/GPUWriteRead_cuda
```

**Adios Kokkos code**

Code that makes calls to ADIOS from within a Kokkos code. Same example as Cuda Adios the writes alternate between CPU and GPU but this time the GPU data is also updated in every simulation step. 

```
export PATH=$PATH:/usr/local/cuda-11.1/bin
mkdir build
cd build
cmake -D Kokkos_ROOT=/home/gainaru/kokkos/install/lib/cmake -D adios2_ROOT=/home/gainaru/adios/install/lib/cmake ../
make -j
./adios.kokkos
Steps expected by the reader: 10
Expecting data per step: 100 elements
Simualation step 0 : 100 elements: 5
Simualation step 1 : 100 elements: 50
Simualation step 2 : 100 elements: 25
Simualation step 3 : 100 elements: 150
Simualation step 4 : 100 elements: 45
Simualation step 5 : 100 elements: 250
Simualation step 6 : 100 elements: 65
Simualation step 7 : 100 elements: 350
Simualation step 8 : 100 elements: 85
Simualation step 9 : 100 elements: 450
```

## Installing Nvidia drivers

Installing and configuring NVIDIA GPUDirect Storage on sdg-tm76

**Software** <br/>
Ubuntu 18.04 <br/>
CUDA v11.1 with NVIDIA driver v456: [Cuda downloads](https://developer.nvidia.com/cuda-downloads)<br/>
GDS 0.9: [GPU Direct Storage downloads](https://developer.nvidia.com/gpudirect-storage-open-beta-v09-r11-1-ubuntu-1804) <br/>
MOFED 5.1: [Mellanox OFED](https://www.mellanox.com/products/infiniband-drivers/linux/mlnx_ofed) <br/>

**Installation Instruction** <br/>
Official instruction: [Release notes](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html) <br/>
MOFED is required to install when NVME is used even if we don’t need RDMA over IB. <br/>
Installation order: `MOFED -> CUDA with NVIDIA driver -> GDS`. Note that GDS may wrongly uninstall NVIDIA driver, so you may need to install the NVIDIA driver again in the end. <br/>
GDS will force you to use the latest NVIDIA driver (v455), so don’t try to use a lower version. <br/>

**Filesystem configuration** <br/>
Besides the configuration mentioned in the official instruction, the EXT4 filesystem needs to be configured with ‘data=ordered’ mounting option as instructed [here](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#mount-ext4-fs) <br/>
On sdg-tm76, this only works for the second NVME `/dev/nvme1n1p1` with manual mount operations. This means it won’t work when configuring `/etc/fstab`. You need to manually unmount (sudo umount /mnt/nvme) and mount with correct options (`sudo mount -o data=ordered /dev/nvme1n1p1 /mnt/nvme`). No reboot is required.<br/>
After that the directory /mnt/nvme should be able to be written and read by GPU directly.</br>

**GPU to be used** <br/>
There are 4 GPUs on the system. Device ID 0-2 are RTX 2080 Ti, which are gaming GPUs and do not support GDS. The last one with device ID=3 is a Quadro RTX 5000 GPU which does support GDS. You need to configure that in your program to only use that GPU for I/O.

**Debugging GDS** <br/>
To enable outputting trace information for debugging, toggle the `level` option in `/etc/cufile.json` to be `TRACE`. Then, a file `cufile.log` will be generated containing trace information.
