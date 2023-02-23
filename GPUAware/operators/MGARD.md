# MGARD on Summit

## Pre-requisite

```
module load gcc cmake cuda
```

## Protobuf

Install protocolbuffers (not available on Summit): [https://github.com/protocolbuffers/protobuf](https://github.com/protocolbuffers/protobuf)

```bash
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git submodule update --init --recursive
mkdir build
cmake -DCMAKE_INSTALL_PREFIX={install path} -Dprotobuf_BUILD_SHARED_LIBS=ON ..
make -j4
make install
```

The installed folder will contain `lib64/cmake`

On system76, protobuf is alread installed, but the path to it needs to be added in the `LD_LIBRARY_PATH`

```bash
export LD_LIBRARY_PATH=/usr/local/lib
```

## Install NVComp

Download from:
```
  wget https://developer.download.nvidia.com/compute/nvcomp/2.4.1/local_installers/nvcomp_2.4.1_x86_64_11.x.tgz
```

## Install MGARD

MGARD can be downloaded from: [https://github.com/CODARcode/MGARD](https://github.com/CODARcode/MGARD)

```bash
git clone https://github.com/CODARcode/MGARD.git
mkdir build; cd build
cmake -DCMAKE_PREFIX_PATH=/ccs/home/againaru/adios/mgard/protobuf/install -DCMAKE_INSTALL_PREFIX=/ccs/home/againaru/adios/mgard/MGARD/install ..
make -j4
make install
```

## Install ADIOS with MGARD

```
module load zstd
cmake -D CMAKE_CXX_COMPILER=g++ -D CMAKE_C_COMPILER=gcc -D MGARD_ROOT=/ccs/home/againaru/adios/mgard/MGARD/install ..
make -j4
```

# Running a simple example

On Summit, build both ADIOS2 and MGARD
```
bash build_mgard_cuda_summit.sh
bash build_adios_cuda_mgard.sh
```

Build the example
```
cmake -DADIOS2_ROOT=/path/to/ADIOS2/install ..
make -j4
./MGARDWriteRead

```
