# GPU - Storage direct access

## Steps to run Nvidia GDS code

Documentation about the NVidia cuFile API can be found [here](https://docs.nvidia.com/gpudirect-storage/index.html).

Instructions to build Nvidia cuFile code are for sdg-tm76 at ORNL. Configuring and installing the Nvidia driver on sdg-tm76 can be found [here](https://docs.google.com/document/d/1j___qra3mpecBxy_J9MKi38wKoQafC4J3oKcDyhcEMw)

Exampes with code using the cuFile API can be found at `/usr/local/cuda-11.1/gds/samples`

Currently nvcc is not in `PATH` so it need to be added before compiling any GPU code:
```
export PATH=$PATH:/usr/local/cuda-11.1/bin
```

Flags needed for compile:
```
 g++ -Wall -I /usr/local/cuda/include/  -I /usr/local/cuda-11.1/.targets/x86_64-linux/lib/ hello_cuda.cc -o hello_cuda -L /usr/local/cuda-11.1/targets/x86_64-linux/lib/ -lcufile -L /usr/local/cuda/lib64/ -lcuda -L /usr/local/cuda/lib64/ -lcudart_static -lrt -lpthread -ldl

```

Compile the code
```
make KOKKOS_DEVICES=Cuda
```
