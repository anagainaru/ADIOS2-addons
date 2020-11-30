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
