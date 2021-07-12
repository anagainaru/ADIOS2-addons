# GPU aware ADIOS

**Goal of this research:** Use GPU buffers directly with `Put` functions and save one `memcpy` to the ADIOS internal buffers (illustrated in the following figures). 

<img width="750" alt="GPU aware ADIOS" src="https://user-images.githubusercontent.com/16229479/125330323-9b57b680-e314-11eb-93b2-8c6afec1a0e7.png">

The interface is similar to the classical ADIOS (and does not change at all when CPU buffers are being used).
```c++
// Creating the engine and Variables (not changed)
adios2::Engine bpWriter;
auto data = io.DefineVariable<float>("data", shape, start, count);

// Put CPU buffers (not changed)
bpWriter.Put(data, cpuData);

// Put GPU buffers requires setting a memory space
data.SetMemorySpace(adios2::MemorySpace::CUDA);
bpWriter.Put(data, gpuData);
```

Available memory spaces are `Host` (default), `CUDA` and `Detect`. The `Detect` memory space askes ADIOS to attempt to detect automatically the provenance of the buffer (in case of failure, Host will be assumed). Currently ADIOS supports detecting buffers allocated with CUDA functions. 

**Code**

Changes to the code to allow ADIOS to receive buffers allocated in the GPU memory space in the Put function. Code is stored in the https://github.com/anagainaru/ADIOS2 repo in branch `gpu_copy_to_host`. Description of the changes can be found in this folder, here: [code_changes.md](https://github.com/anagainaru/ADIOS2-addons/blob/main/GPUAware/ADIOS.cuda/code_changes.md)

**Performance**

Overhead of checking the buffer allocation:

```
CPU STD vector: 5199 to 6084 nanoseconds (5-6 microseconds)
CPU buffers allocated with CUDA: 1159 to 2184 nanoseconds (1-2 microseconds)
GPU buffers allocated with CUDA: 1137 to 2291 nanoseconds (1-2 microseconds)
```

ADIOS Initial performance results

<img width="500" alt="GPU aware ADIOS results" src="https://user-images.githubusercontent.com/16229479/120683014-1eab0000-c46b-11eb-8ff7-e799fa2db552.png">
