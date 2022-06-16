
## Link ADIOS with Kokkos 

Similar to linking CUDA, the code changes find the Kokkos package and link it with the `adios2_core` executable.

```diff
diff --git a/CMakeLists.txt b/CMakeLists.txt
index d8bf46fcb..fe70824b7 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -146,6 +146,7 @@ adios_option(LIBPRESSIO "Enable support for LIBPRESSIO transforms" AUTO)
 adios_option(MGARD      "Enable support for MGARD transforms" AUTO)
 adios_option(PNG        "Enable support for PNG transforms" AUTO)
 adios_option(CUDA       "Enable support for Cuda" AUTO)
+adios_option(Kokkos     "Enable support for Kokkos" AUTO)
 adios_option(MPI        "Enable support for MPI" AUTO)
 adios_option(DAOS       "Enable support for DAOS" AUTO)
 adios_option(DataMan    "Enable support for DataMan" AUTO)
 @@ -222,7 +223,7 @@ endif()


 set(ADIOS2_CONFIG_OPTS
-    BP5 DataMan DataSpaces HDF5 HDF5_VOL MHS SST CUDA Fortran MPI Python Blosc BZip2 LIBPRESSIO MGARD PNG SZ ZFP DAOS IME O_DIRECT Sodium SysVShMem ZeroMQ Profiling Endian_Reverse
+    BP5 DataMan DataSpaces HDF5 HDF5_VOL MHS SST Fortran MPI Python Blosc BZip2 LIBPRESSIO MGARD PNG SZ ZFP CUDA Kokkos DAOS IME O_DIRECT Sodium SysVShMem ZeroMQ Profiling Endian_Reverse
 )

 GenerateADIOSHeaderConfig(${ADIOS2_CONFIG_OPTS})
```

```diff
diff --git a/cmake/DetectOptions.cmake b/cmake/DetectOptions.cmake
index d4e2e5dbf..da9300ca1 100644
--- a/cmake/DetectOptions.cmake
+++ b/cmake/DetectOptions.cmake
@@ -180,6 +180,18 @@ if(CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
   set(ADIOS2_HAVE_CUDA TRUE)
 endif()

+# Kokkos
+if(ADIOS2_USE_Kokkos)
+  if(ADIOS2_USE_Kokkos STREQUAL AUTO)
+    find_package(Kokkos QUIET)
+  else()
+    find_package(Kokkos REQUIRED)
+  endif()
+endif()
+if(Kokkos_FOUND)
+  set(ADIOS2_HAVE_Kokkos TRUE)
+endif()
+
 # Fortran
 if(ADIOS2_USE_Fortran STREQUAL AUTO)
   include(CheckLanguage)
   ```
   
   ```diff
   diff --git a/source/adios2/CMakeLists.txt b/source/adios2/CMakeLists.txt
index e8551ff04..45c869fa1 100644
--- a/source/adios2/CMakeLists.txt
+++ b/source/adios2/CMakeLists.txt
@@ -123,6 +123,10 @@ if(ADIOS2_HAVE_CUDA)
   set(maybe_adios2_core_cuda adios2_core_cuda)
 endif()

+if(ADIOS2_HAVE_Kokkos)
+    target_link_libraries(adios2_core PRIVATE Kokkos::kokkos)
+endif()
+
 target_include_directories(adios2_core
   PUBLIC
     $<BUILD_INTERFACE:${ADIOS2_SOURCE_DIR}/source>
   ```
   
   Adding the Kokkos backend in ADIOS
   ```diff
   diff --git a/source/adios2/common/ADIOSTypes.h b/source/adios2/common/ADIOSTypes.h
index 03cbc19d1..7e2750717 100644
--- a/source/adios2/common/ADIOSTypes.h
+++ b/source/adios2/common/ADIOSTypes.h
@@ -37,7 +37,8 @@ enum class MemorySpace
 {
     Detect, ///< Detect the memory space automatically
     Host,   ///< Host memory space (default)
-    CUDA    ///< CUDA memory spaces
+    CUDA,   ///< CUDA memory spaces
+    Kokkos  ///< Kokkos memory space
 };

 /** Variable shape type identifier, assigned automatically from the signature of
   ```
   
  ## Building ADIOS with Kokkos
  
  ```bash
  cmake -DKokkos_ROOT=/path/to/kokkos/install -DADIOS2_USE_Kokkos=ON  ../ADIOS2/
  ```
  
