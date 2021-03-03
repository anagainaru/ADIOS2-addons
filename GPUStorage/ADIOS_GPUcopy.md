# ADIOS with GPU buffers

Changes to the code to allow ADIOS to receive buffers allocated in the GPU memory space in the `Put` function.
Code is stored in the `https://github.com/anagainaru/ADIOS2` repo in branch `gpu_copy_to_host`.

Currently only Cuda is supported.

## Detect the CUDA environment

Changes in `${ADIOS_ROOT}/CMakeLists` and `${ADIOS_ROOT}/cmake/DetectOptions.cmake`.
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

## Detect host/device buffers

If ADIOS was build with Cuda and the mode is Sync, each buffer submitted to ADIOS is inspected.
If the buffer was allocated on the device, ADIOS will copy it to host and store the data in its internal buffers.

```diff
diff --git a/source/adios2/core/Engine.tcc b/source/adios2/core/Engine.tcc
index f2540258c..38aa06353 100644
--- a/source/adios2/core/Engine.tcc
+++ b/source/adios2/core/Engine.tcc
@@ -17,6 +17,12 @@
 
 #include "adios2/helper/adiosFunctions.h" // CheckforNullptr
 
+#ifdef ADIOS2_HAVE_CUDA
+  #include <cuda.h>
+  #include <cuda_runtime.h>
+  #include "cufile.h"
+#endif
+
 namespace adios2
 {
 namespace core
@@ -39,6 +45,19 @@ typename Variable<T>::Span &Engine::Put(Variable<T> &variable,
 template <class T>
 void Engine::Put(Variable<T> &variable, const T *data, const Mode launch)
 {
+    #ifdef ADIOS2_HAVE_CUDA
+        size_t count = helper::GetTotalSize(variable.Count());
+        std::vector<T> hostData(count);
+	    cudaPointerAttributes attributes;
+	    cudaPointerGetAttributes(&attributes, (const void *) data);
+	    if(attributes.devicePointer != NULL){
+	        // if the buffer is on GPU memory copy it to the CPU
+	        cudaMemcpy(hostData.data(), data, count * sizeof(T),
+		           cudaMemcpyDeviceToHost);
+	        data = hostData.data();
+	    }
+    #endif
+
     CommonChecks(variable, data, {{Mode::Write, Mode::Append}},
                  "in call to Put");
```

```diff
diff --git a/source/adios2/CMakeLists.txt b/source/adios2/CMakeLists.txt
index 2b3a84447..b1f90830f 100644
--- a/source/adios2/CMakeLists.txt
+++ b/source/adios2/CMakeLists.txt
@@ -115,6 +115,13 @@ add_library(adios2_core
 set_property(TARGET adios2_core PROPERTY EXPORT_NAME core)
 set_property(TARGET adios2_core PROPERTY OUTPUT_NAME adios2${ADIOS2_LIBRARY_SUFFIX}_core)
 
+if(ADIOS2_HAVE_CUDA)
+  target_include_directories(adios2_core PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda-11.1/targets/x86_64-linux/lib/)
+  target_link_directories(adios2_core PUBLIC /usr/local/cuda-11.1/targets/x86_64-linux/lib/)
+  target_link_libraries(adios2_core PUBLIC ${CUDA_LIBRARIES} -lcufile)
+  #message (FATAL_ERROR "${CUDA_LIBRARIES}")
+endif()
+
 target_include_directories(adios2_core
   PUBLIC
     $<BUILD_INTERFACE:${ADIOS2_SOURCE_DIR}/source>
```

## Add an example using GPU buffers

```diff
diff --git a/examples/CMakeLists.txt b/examples/CMakeLists.txt
index 77f5a3844..50a8f29a4 100644
--- a/examples/CMakeLists.txt
+++ b/examples/CMakeLists.txt
@@ -7,6 +7,7 @@ add_subdirectory(basics)
 add_subdirectory(useCases)
+add_subdirectory(gpuDirect)
 
 if(ADIOS2_HAVE_MPI)
   add_subdirectory(heatTransfer)
```

The example is similar to the `bpRead`, `bpWrite` examples, except it uses buffers allocated on the GPU and kernels for computations.

```diff
diff --git a/examples/gpuDirect/bpWriteRead.cpp b/examples/gpuDirect/bpWriteRead.cpp
new file mode 100644
+++ b/examples/gpuDirect/bpWriteRead.cpp
@@ -0,0 +1,113 @@
+  #include <cuda.h>
+  #include <cuda_runtime.h>
+
+    float *gpuSimData;
+    cudaMalloc(&gpuSimData, N);
+    cudaMemset(gpuSimData, 0, N);
+ 
+        bpWriter.BeginStep();
+	    bpWriter.Put(data, gpuSimData);
+        bpWriter.EndStep();
```

