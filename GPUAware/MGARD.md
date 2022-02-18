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
mkdir build
cmake ../cmake/ -DCMAKE_INSTALL_PREFIX={install path} -Dprotobuf_BUILD_SHARED_LIBS=ON
make -j4
make install
```

The installed folder will contain `lib64/cmake`

On system76, protobuf is alread installed, but the path to it needs to be added in the `LD_LIBRARY_PATH`

```bash
export LD_LIBRARY_PATH=/usr/local/lib
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
cmake -D CMAKE_CXX_COMPILER=g++ -D CMAKE_C_COMPILER=gcc -D MGARD_LIBRARY=/ccs/home/againaru/adios/mgard/MGARD/install/lib64 -D MGARD_INCLUDE_DIR=/ccs/home/againaru/adios/mgard/MGARD/install/include ..
make -j4
```
