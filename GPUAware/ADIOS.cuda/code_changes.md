# ADIOS with GPU buffers

Changes to the code to allow ADIOS to receive buffers allocated in the GPU memory space in the `Put` function.
Code is stored in the `https://github.com/anagainaru/ADIOS2` repo in branch `gpu_copy_to_host`.

Currently only `CUDA` is supported.

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

### Build cuda specific files

Cuda specific files are located in `source/adios2/helper/adiosCUDA.cu`

```diff
diff --git a/source/adios2/CMakeLists.txt b/source/adios2/CMakeLists.txt
index 3cc0d4ed2..82a4e8136 100644
--- a/source/adios2/CMakeLists.txt
+++ b/source/adios2/CMakeLists.txt
@@ -48,6 +48,7 @@ add_library(adios2_core
   helper/adiosXML.cpp
+  helper/adiosCUDA.cu

 #engine derived classes
   engine/bp3/BP3Reader.cpp engine/bp3/BP3Reader.tcc
@@ -116,8 +117,10 @@ set_property(TARGET adios2_core PROPERTY OUTPUT_NAME adios2${ADIOS2_LIBRARY_SUFFIX}_core)

 if(ADIOS2_HAVE_CUDA)
+  enable_language(CUDA)
   target_include_directories(adios2_core PUBLIC ${CUDA_INCLUDE_DIRS})
   target_link_libraries(adios2_core PUBLIC ${CUDA_LIBRARIES})
+  set_target_properties(adios2_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 endif()
```

## Manage GPU buffers inside ADIOS

### List of memory spaces for buffers allocation

Define the accepted memory spaces in `source/adios2/common/ADIOSTypes.h`. For now only `CUDA` is supported. By default the `Host` memory space is chosen.

```diff
diff --git a/source/adios2/common/ADIOSTypes.h b/source/adios2/common/ADIOSTypes.h
index 8f4a9ae49..7309f5e4b 100644
--- a/source/adios2/common/ADIOSTypes.h
+++ b/source/adios2/common/ADIOSTypes.h
@@ -64,6 +64,15 @@ enum class Mode
 };

+/** Memory space for the buffers received with Put */
+enum class MemorySpace
+{
+    Detect, ///< detect automatically
+    Host, ///< default memory space
+    CUDA  ///< GPU memory spaces
+};
+
 enum class ReadMultiplexPattern
 {
```

The `Detect` memory space can be used to allow ADOIS to automatically detect where the buffer has been allocated.

### Add a MemorySpace to an ADIOS variables

**1. Create an IsGPU field within each variable**

Variables will contain an extra field `m_MemorySpace` set by default to `Host` that can be updated when setting the `CUDA` memory space. Buffer independent functions are implemented in `VariableBase.*`.

```diff
diff --git a/source/adios2/core/VariableBase.h b/source/adios2/core/VariableBase.h
index 19a467b..58447bf 100644
--- a/source/adios2/core/VariableBase.h
+++ b/source/adios2/core/VariableBase.h
@@ -46,7 +46,7 @@ public:
      *  VariableCompound -> from constructor sizeof(struct) */
     const size_t m_ElementSize;
+    MemorySpace m_MemorySpace = MemorySpace::Host;

     ShapeID m_ShapeID = ShapeID::Unknown; ///< see shape types in ADIOSTypes.h
@@ -125,7 +130,7 @@ public:
   size_t TotalSize() const noexcept;

+  /**
+   * Set the memory space
+   * @param the memory space where the expected buffers were allocated
+   */
+  void SetMemorySpace(const MemorySpace mem);

   /**
diff --git a/source/adios2/core/VariableBase.cpp b/source/adios2/core/VariableBase.cpp
index 5996587..b0753c6 100644
--- a/source/adios2/core/VariableBase.cpp
+++ b/source/adios2/core/VariableBase.cpp
@@ -45,14 +45,7 @@ size_t VariableBase::TotalSize() const noexcept

+ void VariableBase::SetMemorySpace(const MemorySpace mem)
+ {
+     m_MemorySpace = mem;
+ }

 void VariableBase::SetShape(const adios2::Dims &shape)
```

The `InfoBlocks` within a Variable have an extra field `m_IsGPU` updated to True if the buffer has been allocated on the GPU.
The variable's vallue is set based on the MemorySpace provided by the user for a given Variable. If the MemorySpace is set to `Detect`, ADIOS will automatically check where the buffer has been allocated and set the `m_IsGPU` variable accordingly. 

```diff
diff --git a/source/adios2/core/Variable.h b/source/adios2/core/Variable.h
index 2bb5a64f1..1e27868b2 100644
--- a/source/adios2/core/Variable.h
+++ b/source/adios2/core/Variable.h
@@ -71,6 +71,7 @@ public:
     T m_Value = T();
+    bool IsBufferOnGPU(const T* data) const;

     struct Info
@@ -106,6 +107,7 @@ public:
         bool IsReverseDims = false;
+        bool IsGPU = false;
     };

     /** use for multiblock info */
```

For every new `BlockInfo` the `IsGPU` flag is set based on information at the Variable level. 
```diff
diff --git a/source/adios2/core/Variable.cpp b/source/adios2/core/Variable.cpp
index a37c60c7f..e33d8810d 100644
--- a/source/adios2/core/Variable.cpp
+++ b/source/adios2/core/Variable.cpp
@@ -49,6 +63,7 @@ namespace core
         info.Operations = m_Operations;                                        \
+        info.IsGPU = IsBufferOnGPU(data);                                      \
         m_BlocksInfo.push_back(info);                                          \
     }                                                                          \
diff --git a/source/adios2/core/Variable.tcc b/source/adios2/core/Variable.tcc
index ccfc5a4..401cdcc 100644
--- a/source/adios2/core/Variable.tcc
+++ b/source/adios2/core/Variable.tcc
@@ -22,6 +22,26 @@ namespace core
 {

+template <class T>
+bool Variable<T>::IsBufferOnGPU(const T* data) const
+{
+    if( m_MemorySpace == MemorySpace::CUDA )
+        return true;
+    if( m_MemorySpace == MemorySpace::Host )
+        return false;
+
+    #ifdef ADIOS2_HAVE_CUDA
+    cudaPointerAttributes attributes;
+    cudaError_t status = cudaPointerGetAttributes(
+        &attributes, (const void *) data);
+    if (status != 0)
+        return false;
+    if(attributes.devicePointer != NULL)
+        return true;
+    #endif
+    return false;
+}
+
 template <class T>
 Dims Variable<T>::DoShape(const size_t step) const
 {
     CheckRandomAccess(step, "Shape");
```

**2. Update CXX bindings to propagate info about the MemorySpace**

For C++ codes using ADIOS, update the `Variable.{c|h}` functions to include the `SetMemorySpace` function.

```diff
diff --git a/bindings/CXX11/adios2/cxx11/Variable.h b/bindings/CXX11/adios2/cxx11/Variable.h
index c1928f47c..63afadfa8 100644
--- a/bindings/CXX11/adios2/cxx11/Variable.h
+++ b/bindings/CXX11/adios2/cxx11/Variable.h
@@ -196,6 +196,11 @@ public:
     void SetStepSelection(const adios2::Box<size_t> &stepSelection);

+    /**
+     * Sets the memory step for all following Puts
+     */
+    void SetMemorySpace(const MemorySpace mem);
+
     /**
      * Returns the number of elements required for pre-allocation based on
diff --git a/bindings/CXX11/adios2/cxx11/Variable.cpp b/bindings/CXX11/adios2/cxx11/Variable.cpp
index fdbccb525..c55c76597 100644
--- a/bindings/CXX11/adios2/cxx11/Variable.cpp
+++ b/bindings/CXX11/adios2/cxx11/Variable.cpp
@@ -32,6 +32,14 @@ namespace adios2
}

+     template <>                                                               \
+    void Variable<T>::SetMemorySpace(const MemorySpace mem)                    \
+    {                                                                          \
+        helper::CheckForNullptr(m_Variable,                                    \
+                                "in call to Variable<T>::SetShape");           \
+        m_Variable->SetMemorySpace(mem);                                       \
+    }                                                                          \
                                                                                \
     template <>                                                                \
     void Variable<T>::SetShape(const Dims &shape)                              \
```

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
