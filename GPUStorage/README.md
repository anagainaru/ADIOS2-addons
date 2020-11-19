# GPU - Storage direct access

## Steps to run Nvidia GDS code

Documentation about the NVidia cuFile API can be found [here](https://docs.nvidia.com/gpudirect-storage/index.html).

Instructions to build Nvidia cuFile code are for System76 at ORNL.

Exampes with code using the cuFile API can be found at `/usr/local/cuda-11.1/gds/samples`

Flags needed for compile:
```
CUDA_PATH   := /usr/local/cuda
CUFILE_PATH := /usr/local/cuda-11.1/targets/x86_64-linux/lib/
CXXFLAGS    := -Wall
CXXFLAGS    += -I $(CUDA_PATH)/include/
CXXFLAGS    += -I $(CUFILE_PATH)


```
