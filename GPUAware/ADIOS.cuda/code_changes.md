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
 
 diff --git a/cmake/DetectOptions.cmake b/cmake/DetectOptions.cmake
index 94ce684e9..78a50e3e5 100644
--- a/cmake/DetectOptions.cmake
+++ b/cmake/DetectOptions.cmake
@@ -143,9 +143,9 @@ set(mpi_find_components C)
   
+# Cuda
+if(ADIOS2_USE_CUDA STREQUAL AUTO)
+  find_package(CUDAToolkit)
+elseif(ADIOS2_USE_CUDA)
+  find_package(CUDAToolkit REQUIRED)
+endif()
+if(CUDAToolkit_FOUND)
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
+  target_link_libraries(adios2_core PUBLIC CUDA::cudart CUDA::cuda_driver)
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
   target_link_libraries(adios2_core PUBLIC CUDA::cudart CUDA::cuda_driver)
+  set_target_properties(adios2_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 endif()
```

## Manage GPU buffers inside ADIOS

<img width="500" alt="ADIOS CUDA aware implementation" src="https://user-images.githubusercontent.com/16229479/125333190-f17a2900-e317-11eb-9fd0-69722cdf0e06.png">

ADIOS variables keeps a memory space (by default Host). Each blockInfo within a Variable has a flag
	`info.IsGPU = IsBufferOnGPU(data)`. The ADIOS Variable implements IsBufferOnGPU(data) based on the memory space. If memory space is set to `Detect`, ADIOS detects automatically the provenance, otherwise trusts the user
 
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
+    MemorySpace m_MemorySpace = MemorySpace::Detect;

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
+        info.IsGPU = IsCUDAPointer((void *) data);                             \
         m_BlocksInfo.push_back(info);                                          \
     }                                                                          \
diff --git a/source/adios2/core/Variable.tcc b/source/adios2/core/Variable.tcc
index ccfc5a4..401cdcc 100644
--- a/source/adios2/core/VariableBase.cpp
+++ b/source/adios2/core/VariableBase.cpp
@@ -22,6 +22,26 @@ namespace core
    InitShapeType();
}

+ bool VariableBase::IsCUDAPointer(void *ptr)
+ {
+    if( m_MemorySpace == MemorySpace::CUDA )
+        return true;
+    if( m_MemorySpace == MemorySpace::Host )
+        return false;
+
+    #ifdef ADIOS2_HAVE_CUDA
+    cudaPointerAttributes attr;
+    cudaPointerGetAttributes(&attr, ptr);
+    return attr.type == cudaMemoryTypeDevice;
+    #endif
+
+    return false;
+ }

size_t VariableBase::TotalSize() const noexcept
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
+        m_Variable->SetMemorySpace(mem);                                       \
+    }                                                                          \
                                                                                \
     template <>                                                                \
     void Variable<T>::SetShape(const Dims &shape)                              \
```
