# Get with GPU buffers

Interface
```c++
data.SetMemorySpace(adios2::MemorySpace::CUDA);
bpWriter.Get(data, gpuData);
```

This file describes the changes to the ADIOS library specific to the Get function (inside BP4). All changes described in the [code_changes.md](https://github.com/anagainaru/ADIOS2-addons/blob/main/GPUAware/ADIOS.cuda/code_changes.md) file are required.

### Add a copy function from the ADIOS buffer to the GPU

### Add an example using GPU buffers

