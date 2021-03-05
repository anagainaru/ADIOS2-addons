# ADIOS with GPU buffers

Changes to the code to allow ADIOS to receive buffers allocated in the GPU memory space in the `Put` function.
Code is stored in the `https://github.com/anagainaru/ADIOS2` repo in branch `gpu_copy_to_host`.

Currently only Cuda is supported.

## Link ADIOS with CUDA

By detecting the CUDA environment and linking it with all the functions implemented by ADIOS.

### Detect the CUDA environment

Changes in `${ADIOS_ROOT}/CMakeLists` and `${ADIOS_ROOT}/cmake/DetectOptions.cmake` to detect the Cuda compiler during build.

```diff
diff --git a/CMakeLists.txt b/CMakeLists.txt
index 5d327a8ba..458ef87c8 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -120,6 +120,7 @@ adios_option(MGARD     "Enable support for MGARD transforms" AUTO)
 adios_option(PNG       "Enable support for PNG transforms" AUTO)
+adios_option(CUDA       "Enable support for Cuda" AUTO)
 adios_option(MPI       "Enable support for MPI" AUTO)
@@ -147,7 +148,7 @@ if(ADIOS2_HAVE_MPI)
 endif()
 
 set(ADIOS2_CONFIG_OPTS
-    Blosc BZip2 ZFP SZ MGARD PNG MPI DataMan Table SSC SST DataSpaces ZeroMQ HDF5 IME Python Fortran SysVShMem Profiling Endian_Reverse
+  Blosc BZip2 ZFP SZ MGARD PNG CUDA MPI DataMan Table SSC SST DataSpaces ZeroMQ HDF5 IME Python Fortran SysVShMem Profiling Endian_Reverse
 )
 GenerateADIOSHeaderConfig(${ADIOS2_CONFIG_OPTS})
 configure_file(
@@ -280,6 +281,9 @@ message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
                          "${CMAKE_CXX_COMPILER_WRAPPER}")
 message("    ${CMAKE_CXX_COMPILER}")
 message("")
+if(ADIOS2_HAVE_CUDA)
+	message("  Cuda Compiler : ${CMAKE_CUDA_COMPILER} ")
+endif()
 if(ADIOS2_HAVE_Fortran)
   message("  Fortran Compiler : ${CMAKE_Fortran_COMPILER_ID} "
                                "${CMAKE_Fortran_COMPILER_VERSION} "
```

```diff
diff --git a/cmake/DetectOptions.cmake b/cmake/DetectOptions.cmake
index ca449feee..38c1f9cee 100644
--- a/cmake/DetectOptions.cmake
+++ b/cmake/DetectOptions.cmake
@@ -129,6 +129,16 @@ endif()
 
 set(mpi_find_components C)
 
+# Cuda
+if(ADIOS2_USE_CUDA STREQUAL AUTO)
+  find_package(CUDA)
+elseif(ADIOS2_USE_CUDA)
+  find_package(CUDA REQUIRED)
+endif()
+if(CUDA_FOUND)
+  set(ADIOS2_HAVE_CUDA TRUE)
+endif()
+
 # Fortran
 if(ADIOS2_USE_Fortran STREQUAL AUTO)
   include(CheckLanguage)
```

### Link the adios_core library to CUDA

If CUDA is detected on the system, link CUDA directly to the entire functions defined by ADIOS.

```diff
diff --git a/source/adios2/CMakeLists.txt b/source/adios2/CMakeLists.txt
index 2b3a84447..b1f90830f 100644
--- a/source/adios2/CMakeLists.txt
+++ b/source/adios2/CMakeLists.txt
@@ -115,6 +115,13 @@ add_library(adios2_core
 set_property(TARGET adios2_core PROPERTY EXPORT_NAME core)
 set_property(TARGET adios2_core PROPERTY OUTPUT_NAME adios2${ADIOS2_LIBRARY_SUFFIX}_core)
 
+if(ADIOS2_HAVE_CUDA)
+  target_include_directories(adios2_core PUBLIC ${CUDA_INCLUDE_DIRS})
+  target_link_libraries(adios2_core PUBLIC ${CUDA_LIBRARIES})
+endif()
+
 target_include_directories(adios2_core
   PUBLIC
     $<BUILD_INTERFACE:${ADIOS2_SOURCE_DIR}/source>
```

## Manage GPU buffers inside ADIOS

### Detect the GPU buffer

Detect for each `Put` function and mark the buffer as allocated on the device.

```diff
diff --git a/source/adios2/core/Variable.cpp b/source/adios2/core/Variable.cpp
index a37c60c7f..f59330f60 100644
--- a/source/adios2/core/Variable.cpp
+++ b/source/adios2/core/Variable.cpp
@@ -49,6 +49,7 @@ namespace core
         info.StepsCount = stepsCount;                                          \
         info.Data = const_cast<T *>(data);                                     \
         info.Operations = m_Operations;                                        \
+        info.IsGPU = IsBufferOnGPU(const_cast<T *>(data));                     \
         m_BlocksInfo.push_back(info);                                          \
         return m_BlocksInfo.back();                                            \
     }                                                                          \
diff --git a/source/adios2/core/Variable.h b/source/adios2/core/Variable.h
index 2bb5a64f1..a3e5b5f6c 100644
--- a/source/adios2/core/Variable.h
+++ b/source/adios2/core/Variable.h
@@ -106,6 +106,7 @@ public:
         SelectionType Selection = SelectionType::BoundingBox;
         bool IsValue = false;
         bool IsReverseDims = false;
+        bool IsGPU = false;
     };
@@ -147,6 +148,8 @@ public:
     AllStepsBlocksInfo() const;

 private:
+    bool IsBufferOnGPU(const T* data);
+
     Dims DoShape(const size_t step) const;

     Dims DoCount() const;
diff --git a/source/adios2/core/Variable.tcc b/source/adios2/core/Variable.tcc
index edf7e9b5c..5362c5437 100644
--- a/source/adios2/core/Variable.tcc
+++ b/source/adios2/core/Variable.tcc
@@ -16,10 +16,25 @@
 #include "adios2/core/Engine.h"
 #include "adios2/helper/adiosFunctions.h"

+#ifdef ADIOS2_HAVE_CUDA
+  #include <cuda.h>
+  #include <cuda_runtime.h>
+#endif
+
 namespace adios2
 {
 namespace core
 {
+template <class T>
+bool Variable<T>::IsBufferOnGPU(const T* data){
+    #ifdef ADIOS2_HAVE_CUDA
+    cudaPointerAttributes attributes;
+    cudaPointerGetAttributes(&attributes, (const void *) data);
+    if(attributes.devicePointer != NULL)
+        return true;
+    #endif
+    return false;
+}

 template <class T>
 Dims Variable<T>::DoShape(const size_t step) const
```
Each `blockInfo` for each `Variable` will store a boolean stating if the `Data` is a host/device buffer.

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

### Manage the device buffer

All `Put` functions eventually call `PutVariableMetadata` and `PutVariablePayload` to copy the information from the user provided array to ADIOS internal buffers. In the first case statistics about the data are being saved, in the second the entire content of the array. The codes need to change to use the GPU equivalent functions for each blockInfo that has an positive `IsGPU` flag.

For the `PutVariableMetadata` function:
```diff
diff --git a/source/adios2/toolkit/format/bp/bp4/BP4Serializer.tcc b/source/adios2/toolkit/format/bp/bp4/BP4Serializer.tcc
index b3b429603..4246dec6e 100644
--- a/source/adios2/toolkit/format/bp/bp4/BP4Serializer.tcc
+++ b/source/adios2/toolkit/format/bp/bp4/BP4Serializer.tcc
@@ -334,6 +334,10 @@ BP4Serializer::GetBPStats(const bool singleValue,
     stats.Step = m_MetadataSet.TimeStep;
     stats.FileIndex = GetFileIndex();

+    if (blockInfo.IsGPU){
+        return stats;
+    }
+
     // support span
     if (blockInfo.Data == nullptr && m_Parameters.StatsLevel > 0)
     {
```

For the `PutVariablePayload` function:
```diff
diff --git a/source/adios2/toolkit/format/bp/BPSerializer.tcc b/source/adios2/toolkit/format/bp/BPSerializer.tcc
index 8414fa6eb..11b127add 100644
--- a/source/adios2/toolkit/format/bp/BPSerializer.tcc
+++ b/source/adios2/toolkit/format/bp/BPSerializer.tcc
@@ -72,6 +72,13 @@ inline void BPSerializer::PutPayloadInBuffer(
 {
     const size_t blockSize = helper::GetTotalSize(blockInfo.Count);
     m_Profiler.Start("memcpy");
+    if(blockInfo.IsGPU){
+        helper::CopyFromGPUToBuffer(m_Data.m_Buffer, m_Data.m_Position,
+                     blockInfo.Data, blockSize);
+        m_Profiler.Stop("memcpy");
+        m_Data.m_AbsolutePosition += blockSize * sizeof(T);
+        return;
+    }
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
