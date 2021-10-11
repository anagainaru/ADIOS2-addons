# Get with GPU buffers

ADIOS does not use an internal buffer and reads data directly in the user provided buffer. In order to use directly GPU buffers, ADIOS would need to allocate memory for a CPU buffer, fill it by reading data and `cudamemcpy` to the user provided GPU buffer. The performance and memory footprint would not be improved. However, GPU Direct to Storage could be used for this case.

Interface
```c++
data.SetMemorySpace(adios2::MemorySpace::CUDA);
bpWriter.Get(data, gpuData);
```

This file describes the changes to the ADIOS library specific to the Get function (inside BP4). All changes described in the [code_changes.md](https://github.com/anagainaru/ADIOS2-addons/blob/main/GPUAware/ADIOS.cuda/code_changes.md) file are required.

### ADIOS workflow for the Get function.

The `Get` function reads payload info by going through the following steps:
- `Get` functions implemented in `adios2/core/engine/Engine.*` files
- `ReadVariableBlocks` called in `../ADIOS2-copy/source/adios2/engine/bp4/BP4Reader.cpp` and defined in ../ADIOS2-copy/source/adios2/engine/bp4/BP4Reader.tcc

```
void BP4Reader::ReadVariableBlocks(Variable<T> &variable)
                m_DataFileManager.ReadFile(buffer, payloadSize, payloadStart,
                                           subStreamBoxInfo.SubStreamID);
```
This is where the buffer allocation would take place.

- The next step uses the POSIX file transport
   - The `ReadFile` defined in `../ADIOS2-copy/source/adios2/toolkit/transportman` calls the defined transport `Read` function
   - The `Read` function in defined in `../ADIOS2-copy/source/adios2/toolkit/transport/file/FilePOSIX.cpp`
```
void FilePOSIX::Read(char *buffer, size_t size, size_t start)
```
For GDS a new transport will be defined and used to bring data to the GPU buffer.

### Add an example using GPU buffers

Changes in `/examples/gpu/cudaWriteRead.cu`

```diff
diff --git a/examples/cuda/cudaWriteRead.cu b/examples/cuda/cudaWriteRead.cu
index e83754ac7..caa7ca0b7 100644
--- a/examples/cuda/cudaWriteRead.cu
+++ b/examples/cuda/cudaWriteRead.cu
@@ -58,6 +58,10 @@ int BPWrite(const std::string fname, const size_t N, int nSteps){
 }

 int BPRead(const std::string fname, const size_t N, int nSteps){
+  float *gpuSimData;
+  cudaMalloc(&gpuSimData, N * sizeof(float));
+  cudaMemset(gpuSimData, 0, N);
+
   // Create ADIOS structures
   adios2::ADIOS adios;
   adios2::IO io = adios.DeclareIO("ReadIO");
@@ -81,8 +85,13 @@ int BPRead(const std::string fname, const size_t N, int nSteps){
   for (size_t step = 0; step < write_step; step++)
   {
       data.SetStepSelection({step, 1});
-      bpReader.Get(data, simData.data());
+      data.SetMemorySpace(adios2::MemorySpace::CUDA);
+      bpReader.Get(data, gpuSimData);
       bpReader.PerformGets();
+
+      cudaMemcpy(simData.data(), gpuSimData, N * sizeof(float),
+              cudaMemcpyDeviceToHost);
+
       std::cout << "Simualation step " << step << " : ";
       std::cout << simData.size() << " elements: " << simData[1] << std::endl;
   }
```
