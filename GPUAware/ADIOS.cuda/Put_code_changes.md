# Put with GPU buffers

Interface
```c++
data.SetMemorySpace(adios2::MemorySpace::CUDA);
bpWriter.Put(data, gpuData);
```

This file describes the changes to the ADIOS library specific to the Put function (inside BP4). All changes described in the [code_changes.md](https://github.com/anagainaru/ADIOS2-addons/blob/main/GPUAware/ADIOS.cuda/code_changes.md) file are required.

### Add a copy function from the GPU to the ADIOS buffer

The ADIOS helper functions that manage memory contains the `CopyToBuffer` function that is used to copy the data from the host to ADIOS buffers.
Similarly we add the `CopyFromGPUToBuffer` function to copy data from the device to ADIOS buffers.

```diff
diff --git a/source/adios2/helper/adiosMemory.h b/source/adios2/helper/adiosMemory.h
index 9d5e40f..3ed6d01 100644
--- a/source/adios2/helper/adiosMemory.h
+++ b/source/adios2/helper/adiosMemory.h
@@ -39,6 +39,15 @@ template <class T>
 void InsertToBuffer(std::vector<char> &buffer, const T *source,
                     const size_t elements = 1) noexcept;

+#ifdef ADIOS2_HAVE_CUDA
+/*
+ * Copies data from a GPU buffer to a specific location in the adios buffer
+ */
+template <class T>
+void CopyFromGPUToBuffer(std::vector<char> &buffer, size_t &position, const T *source,
+                  const size_t elements = 1) noexcept;
+#endif
+
 /**
  * Copies data to a specific location in the buffer updating position
  * Does not update vec.size().
```
The new function `CopyFromGPUToBuffer` is used to copy data from a GPU buffer to a specific location in the adios buffer.

```diff
diff --git a/source/adios2/helper/adiosMemory.inl b/source/adios2/helper/adiosMemory.inl
index f8d450e70..566513755 100644
--- a/source/adios2/helper/adiosMemory.inl
+++ b/source/adios2/helper/adiosMemory.inl
@@ -25,6 +25,11 @@
 #include "adios2/helper/adiosSystem.h"
 #include "adios2/helper/adiosType.h"

+#ifdef ADIOS2_HAVE_CUDA
+  #include <cuda.h>
+  #include <cuda_runtime.h>
+#endif
+
 namespace adios2
 {
 namespace helper
@@ -74,6 +79,19 @@ void InsertToBuffer(std::vector<char> &buffer, const T *source,
     buffer.insert(buffer.end(), src, src + elements * sizeof(T));
 }

+
+#ifdef ADIOS2_HAVE_CUDA
+template <class T>
+void CopyFromGPUToBuffer(std::vector<char> &buffer, size_t &position,
+                         const T *source, const size_t elements) noexcept
+{
+    const char *src = reinterpret_cast<const char *>(source);
+    cudaMemcpy(buffer.data() + position, src, elements * sizeof(T),
+               cudaMemcpyDeviceToHost);
+    position += elements * sizeof(T);
+}
+#endif
+
 template <class T>
 void CopyToBuffer(std::vector<char> &buffer, size_t &position, const T *source,
                   const size_t elements) noexcept
```

### Cuda specific functions for metadata management

Comuting min/max for the data in the GPU buffer will be computed on the GPU using reduction functions provided by the CUDA thrust library.

**Include the cuda specific functions in the adiosFunctions header**
```diff
diff --git a/source/adios2/helper/adiosFunctions.h b/source/adios2/helper/adiosFunctions.h
index 49dc629c6..d689ab035 100644
--- a/source/adios2/helper/adiosFunctions.h
+++ b/source/adios2/helper/adiosFunctions.h
@@ -20,5 +20,6 @@
 #include "adios2/helper/adiosType.h"    //Type casting, conversion, checks, etc.
 #include "adios2/helper/adiosXML.h"     //XML parsing
 #include "adios2/helper/adiosYAML.h"    //YAML parsing
+#include "adios2/helper/adiosCUDA.h"    //CUDA functions

 #endif /* ADIOS2_HELPER_ADIOSFUNCTIONS_H_ */
```

**Compute min/max with CUDA**

Using the thrust library for the reductions. Instantiate the `CUDAMinMax` function for all the datatypes declared in the `ADIOS2_FOREACH_PRIMITIVE_STDTYPE_1ARG` macro (defined in `adios2/common/ADIOSMacros.h`). The thrust reduction function does not work with `complex` datatypes and CUDA does not recognize long double so the `CUDAMinMax` function is overloaded and does not compute min/max for them at this point.

```c++
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include "adios2/common/ADIOSMacros.h"

#include "adiosCUDA.h"

namespace {
template <class T>
void CUDAMinMaxImpl(const T *values, const size_t size, T &min, T &max)
{
    thrust::device_ptr<const T> dev_ptr(values);
    auto res = thrust::minmax_element(dev_ptr, dev_ptr + size);
    cudaMemcpy(&min, thrust::raw_pointer_cast(res.first), sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max, thrust::raw_pointer_cast(res.second), sizeof(T), cudaMemcpyDeviceToHost);
}
// types non supported on the device
void CUDAMinMaxImpl(const long double *values, const size_t size, long double &min, long double &max) {}
void CUDAMinMaxImpl(const std::complex<float> *values, const size_t size, std::complex<float> &min, std::complex<float> &max) {}
void CUDAMinMaxImpl(const std::complex<double> *values, const size_t size, std::complex<double> &min, std::complex<double> &max) {}
}

template <class T>
void adios2::helper::CUDAMinMax(const T *values, const size_t size, T &min, T &max)
{
  CUDAMinMaxImpl(values, size, min, max);
}

#define declare_type(T) \
template void adios2::helper::CUDAMinMax(const T *values, const size_t size, T &min, T &max);
ADIOS2_FOREACH_PRIMITIVE_STDTYPE_1ARG(declare_type)
#undef declare_type
```

### Manage the device buffer

All `Put` functions eventually call `PutVariableMetadata` and `PutVariablePayload` to copy the information from the user provided array to ADIOS internal buffers. In the first case statistics about the data are being saved, in the second the entire content of the array. The codes need to change to use the GPU equivalent functions for each blockInfo that has an positive `IsGPU` flag.

For the **PutVariableMetadata** function:
```diff
diff --git a/source/adios2/toolkit/format/bp/bp4/BP4Serializer.tcc b/source/adios2/toolkit/format/bp/bp4/BP4Serializer.tcc
index 4246dec6e..2345db9e5 100644
--- a/source/adios2/toolkit/format/bp/bp4/BP4Serializer.tcc
+++ b/source/adios2/toolkit/format/bp/bp4/BP4Serializer.tcc
@@ -334,9 +334,13 @@ BP4Serializer::GetBPStats(const bool singleValue,
     stats.Step = m_MetadataSet.TimeStep;
     stats.FileIndex = GetFileIndex();

+   #ifdef ADIOS2_HAVE_CUDA
     if (blockInfo.IsGPU){
+        const size_t size = helper::GetTotalSize(blockInfo.Count);
+        helper::CUDAMinMax(blockInfo.Data, size, stats.Min, stats.Max);
         return stats;
     }
+   #endif

     // support span
```

For the **PutVariablePayload** function:
```diff
diff --git a/source/adios2/toolkit/format/bp/BPSerializer.tcc b/source/adios2/toolkit/format/bp/BPSerializer.tcc
index 8414fa6eb..11b127add 100644
--- a/source/adios2/toolkit/format/bp/BPSerializer.tcc
+++ b/source/adios2/toolkit/format/bp/BPSerializer.tcc
@@ -72,6 +72,13 @@ inline void BPSerializer::PutPayloadInBuffer(
 {
     const size_t blockSize = helper::GetTotalSize(blockInfo.Count);
     m_Profiler.Start("memcpy");
+#ifdef ADIOS2_HAVE_CUDA
+    if(blockInfo.IsGPU){
+        helper::CopyFromGPUToBuffer(m_Data.m_Buffer, m_Data.m_Position,
+                     blockInfo.Data, blockSize);
+        m_Profiler.Stop("memcpy");
+        m_Data.m_AbsolutePosition += blockSize * sizeof(T);
+        return;
+    }
+#endif
     if (!blockInfo.MemoryStart.empty())
     {
         helper::CopyMemoryBlock(
```


## Add an example using GPU buffers

The example will be included in `${ADIOS_ROOT}/examples/gpu`

```diff
diff --git a/examples/CMakeLists.txt b/examples/CMakeLists.txt
index 77f5a3844..50a8f29a4 100644
--- a/examples/CMakeLists.txt
+++ b/examples/CMakeLists.txt
@@ -7,6 +7,7 @@ add_subdirectory(basics)
 add_subdirectory(useCases)
+add_subdirectory(gpu)
 
 if(ADIOS2_HAVE_MPI)
   add_subdirectory(heatTransfer)
```

The example is similar to the `bpRead`, `bpWrite` examples, except it uses buffers allocated on the GPU and kernels for computations.

```diff
diff --git a/examples/gpu/cudaWriteRead.cpp b/examples/gpu/cudaWriteRead.cu
new file mode 100644
+++ b/examples/gpu/cudaWriteRead.cu
@@ -0,0 +1,113 @@
+  #include <cuda.h>
+  #include <cuda_runtime.h>
+
+ __global__ void update_array(int *vect, int val) {
+    vect[blockIdx.x] += val;
+ }
@@ -7,6 +7,7 @@
+ float *gpuSimData;
+ cudaMalloc(&gpuSimData, N);
+ cudaMemset(gpuSimData, 0, N);
+ 
@@ -17,6 +1,113 @@
+ bpWriter.BeginStep();
+ data.SetMemorySpace(adios2::MemorySpace::CUDA);
+	bpWriter.Put(data, gpuSimData);
+ bpWriter.EndStep();
+ update_array<<<N,1>>>(gpuSimData, 1);
```

In order for ADIOS to compile `*.cu` files the cmake file for this example needs to set the target properties to allow `CUDA_SEPARABLE_COMPILATION ON`.

```cmake
if(ADIOS2_HAVE_CUDA)
  enable_language(CUDA)
  add_executable(GPUWriteRead_cuda cudaWriteRead.cu)
  target_include_directories(GPUWriteRead_cuda PUBLIC ${CUDA_INCLUDE_DIRS})
  target_link_libraries(GPUWriteRead_cuda PUBLIC adios2::cxx11 ${CUDA_LIBRARIES})
  set_target_properties(GPUWriteRead_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
```
