```diff
diff --git a/CMakeLists.txt b/CMakeLists.txt
index 5d327a8ba..458ef87c8 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -120,6 +120,7 @@ adios_option(ZFP       "Enable support for ZFP transforms" AUTO)
 adios_option(SZ        "Enable support for SZ transforms" AUTO)
 adios_option(MGARD     "Enable support for MGARD transforms" AUTO)
 adios_option(PNG       "Enable support for PNG transforms" AUTO)
+adios_option(CUDA       "Enable support for Cuda" AUTO)
 adios_option(MPI       "Enable support for MPI" AUTO)
 adios_option(DataMan   "Enable support for DataMan" AUTO)
 adios_option(DataSpaces "Enable support for DATASPACES" AUTO)
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
diff --git a/examples/CMakeLists.txt b/examples/CMakeLists.txt
index 77f5a3844..50a8f29a4 100644
--- a/examples/CMakeLists.txt
+++ b/examples/CMakeLists.txt
@@ -7,6 +7,7 @@ add_subdirectory(basics)
 add_subdirectory(hello)
 add_subdirectory(query)
 add_subdirectory(useCases)
+add_subdirectory(gpuDirect)
 
 if(ADIOS2_HAVE_MPI)
   add_subdirectory(heatTransfer)
diff --git a/examples/gpuDirect/CMakeLists.txt b/examples/gpuDirect/CMakeLists.txt
new file mode 100644
index 000000000..9dd8bff01
--- /dev/null
+++ b/examples/gpuDirect/CMakeLists.txt
@@ -0,0 +1,14 @@
+#------------------------------------------------------------------------------#
+# Distributed under the OSI-approved Apache License, Version 2.0.  See
+# accompanying file Copyright.txt for details.
+#------------------------------------------------------------------------------#
+
+if(ADIOS2_HAVE_CUDA)
+  add_executable(GPUWriteRead_cuda bpWriteRead.cpp)
+  #message (FATAL_ERROR "${CUDA_LIBRARIES} ${CUDA_INCLUDE_DIRS}")
+  target_include_directories(GPUWriteRead_cuda PUBLIC ${CUDA_INCLUDE_DIRS})
+  target_link_libraries(GPUWriteRead_cuda PUBLIC adios2::cxx11 ${CUDA_LIBRARIES})
+else()
+  add_executable(GPUWriteRead bpWriteRead.cpp)
+  target_link_libraries(GPUWriteRead adios2::cxx11)
+endif()
diff --git a/examples/gpuDirect/bpWriteRead.cpp b/examples/gpuDirect/bpWriteRead.cpp
new file mode 100644
index 000000000..06917d9eb
--- /dev/null
+++ b/examples/gpuDirect/bpWriteRead.cpp
@@ -0,0 +1,113 @@
+/*
+ * Simple example of writing and reading data
+ * through ADIOS2 BP engine with multiple simulations steps
+ * for every IO step.
+ */
+
+#include <ios>
+#include <vector>
+#include <iostream>
+
+#include <adios2.h>
+
+#ifdef ADIOS2_HAVE_CUDA
+  #include <cuda.h>
+  #include <cuda_runtime.h>
+#endif
+
+int BPWrite(const std::string fname, const size_t N,
+            int nSteps, float startVal){
+  // Initialize the simulation data
+  int write_step = 10, cpu_step = 1;
+  std::vector<float> simData(N, startVal);
+  #ifdef ADIOS2_HAVE_CUDA
+    cpu_step = 2; // write from cpu every other write step
+    float *gpuSimData;
+    cudaMalloc(&gpuSimData, N);
+    cudaMemset(gpuSimData, startVal, N);
+  #endif
+ 
+  // Set up the ADIOS structures
+  adios2::ADIOS adios;
+  adios2::IO io = adios.DeclareIO("WriteIO");
+
+  // Declare an array for the ADIOS data of size (NumOfProcesses * N)
+  const adios2::Dims shape{static_cast<size_t>(N)};
+  const adios2::Dims start{static_cast<size_t>(0)};
+  const adios2::Dims count{N};
+  auto data = io.DefineVariable<float>("data", shape, start, count);
+
+  adios2::Engine bpWriter = io.Open(fname, adios2::Mode::Write);
+  
+  // Simulation steps
+  for (size_t step = 0; step < nSteps; ++step)
+  {
+      // Make a 1D selection to describe the local dimensions of the
+      // variable we write and its offsets in the global spaces
+      adios2::Box<adios2::Dims> sel({0}, {N});
+      data.SetSelection(sel);
+
+      // Start IO step every write step
+      if (step % write_step == 0){
+        bpWriter.BeginStep();
+        if (step % (write_step * cpu_step) == 0)
+	  bpWriter.Put(data, simData.data());
+	#ifdef ADIOS2_HAVE_CUDA
+          if (step % (write_step * cpu_step) != 0)
+	    bpWriter.Put(data, gpuSimData);
+	#endif
+        bpWriter.EndStep();
+      }
+
+      // Compute new values for the data
+      // for (auto * x: simData) 
+      for (int i = 0; i < N; i++)
+        simData[i] += i;
+  }
+
+  bpWriter.Close();
+  return 0;
+}
+
+int BPRead(const std::string fname, const size_t N, int nSteps){
+  // Create ADIOS structures
+  adios2::ADIOS adios;
+  adios2::IO io = adios.DeclareIO("ReadIO");
+
+  adios2::Engine bpReader = io.Open(fname, adios2::Mode::Read);
+
+  auto data = io.InquireVariable<float>("data");
+  std::cout << "Steps expected by the reader: " << bpReader.Steps() << std::endl;
+  std::cout << "Expecting data per step: " << data.Shape()[0];
+  std::cout  << " elements" << std::endl;
+
+  int write_step = bpReader.Steps();
+  // Create the local buffer and initialize the access point in the ADIOS file
+  std::vector<float> simData(N); //set size to N
+  const adios2::Dims start{0};
+  const adios2::Dims count{N};
+  const adios2::Box<adios2::Dims> sel(start, count);
+  data.SetSelection(sel);
+  
+  // Read the data in each of the ADIOS steps
+  for (size_t step = 0; step < write_step; step++)
+  {
+      data.SetStepSelection({step, 1});
+      bpReader.Get(data, simData.data());
+      bpReader.PerformGets();
+      std::cout << "Simualation step " << step << " : ";
+      std::cout << simData.size() << " elements: " << simData[1] << std::endl;
+  }
+  bpReader.Close();
+  return 0;
+}
+
+int main(int argc, char **argv){
+  const std::string fname("BPAnaWriteRead.bp");
+  const size_t N = 100;
+  int nSteps = 100, ret = 0;
+
+  ret = BPWrite(fname, N, nSteps, 5);
+  ret += BPRead(fname, N, nSteps);
+  return ret;
+}
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
@@ -128,6 +135,17 @@ if(UNIX)
   target_sources(adios2_core PRIVATE toolkit/transport/file/FilePOSIX.cpp)
 endif()
 
+if(ADIOS2_HAVE_CUDA)
+  add_library(adios2_core_gpu
+    toolkit/transport/gpu/GPUdirect.cpp
+  )
+  set_property(TARGET adios2_core_gpu PROPERTY EXPORT_NAME core_gpu)
+  set_property(TARGET adios2_core_gpu PROPERTY OUTPUT_NAME adios2${ADIOS2_LIBRARY_SUFFIX}adios2_core_gpu)
+  target_include_directories(adios2_core_gpu PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda-11.1/targets/x86_64-linux/lib/)
+  target_link_directories(adios2_core_gpu PUBLIC /usr/local/cuda-11.1/targets/x86_64-linux/lib/)
+  target_link_libraries(adios2_core_gpu PUBLIC adios2_core ${CUDA_LIBRARIES} -lcufile)
+endif()
+
 if(ADIOS2_HAVE_MPI)
   add_library(adios2_core_mpi
     core/IOMPI.cpp
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
