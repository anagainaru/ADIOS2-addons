# Running the codes

## Compie the codes

### Summit

The root folder for the experiments using the base engine codes can be found in:
`/gpfs/alpine/csc143/proj-shared/againaru/adios/engine_perf`.

To compile all the codes and create executables for each engine, load all the necessary packages:
```bash
module load cmake
module load gcc
```

Build the codes:
```bash
mkdir build
cd build
cmake -D adios2_ROOT=/ccs/home/againaru/adios/ADIOS2-init/install/lib64/cmake ..
make -j4
```

To run the code, due to incomptibility between libfabrics and mpi.
```
export LD_PRELOAD=/usr/lib64/libibverbs.so.1:/usr/lib64/librdmacm.so.1
```

The codes have been tested with the following versions:
```
cmake version 3.18.2
gcc (GCC) 6.4.0
```

Each node on Summit contains two IBM POWER9 processors and six NVIDIA Tesla V100 accelerators.
One CPU processor contains 12 double precision cores.

### Personal system
```bash
cmake -D adios2_ROOT=~/work/adios/ADIOS2-init/install/lib/cmake ..
make -j4
```

## Simulation scenarios

**Notations:**
- `WR` writers, `RD` readers, `V` variables, `N` array size
- Each writer puts `V * N` floats into the stream and each reader gets `V * N * WR / RD` floats in each run.
- Total amount of data exchanged through the system `D = WR * V * N`

**Basic IO performance**
1. Figure showing how increasing the total exchanged data does not correspond with an increase in execution time / throughput.
2. Figure showing performance as we increase the number of writers / readers
    - Number of writers (`WR = 32, 64, 128, 256, 512, 1024`) and readers `RD = WR / 2`
    - Strong scaling by keeping the same amount of total data (D) and data per writer (D/WR), per reader (D/RD) 
    - Weak scaling by keeping the data per writer fixed, total amount of data increases with WR 
3. Impact of different variables by keeping the total data exchanged constant
    - Impact of number of variables storing the total data (same WR, RD, variables and array size per variable change)
    - Impact of the array dimensions (1D, 2D array)
    - Impact of number of writers putting the total data (same V, RD, writers and array size per writer change)
    - Impact of number of readers getting the total data (same V, WR, readers and array size per reader change)
4. Inline performance

**Basic application performance**
Lipeng scenarios

**Application runs**
XGC with Avocado in different formats


## Code changes

**For one writer multiple readers**

The RendezvousReaderCount parameter needs to be set to 2 in the adios configuration XML file.

```
$ vim adios.xml
<parameter key="RendezvousReaderCount" value="2"/>
```
or the code needs to set the parameters manually
```c++
adios2::Params NoReaders = {{"RendezvousReaderCount", "2"}};
sstIO.SetParameters(NoReaders);
```
Two instances of the reader need to be executed.

path_to_ADIOS2/build/bin/hello_sstWriter & 
path_to_ADIOS2/build/bin/hello_sstReader & 
path_to_ADIOS2/build/bin/hello_sstReader

