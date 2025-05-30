# ADIOS with Kokkos

- [Code changes to allow ADIOS to be build with Kokkos](#code-changes-to-allow-adios-to-be-build-with-kokkos)
- [Code changes to allow Kokkos views in Put and Get](#code-changes-to-allow-kokkos-views-in-put-and-get)


**Building ADIOS with Kokkos** 

Kokkos needs to be build with the memory spaces needed by the application.

  ```bash
  cmake -DKokkos_ROOT=/path/to/kokkos/install -DADIOS2_USE_Kokkos=ON  ../ADIOS2/
  ```

## Code changes to allow ADIOS to be build with Kokkos

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
  
## Code changes to allow Kokkos views in Put and Get
  
  Link the C++ bindings with Kokkos
  
  ```diff
  diff --git a/bindings/CXX11/CMakeLists.txt b/bindings/CXX11/CMakeLists.txt
index 7815a5f33..ddd00eec2 100644
--- a/bindings/CXX11/CMakeLists.txt
+++ b/bindings/CXX11/CMakeLists.txt
@@ -25,6 +25,10 @@ set_property(TARGET adios2_cxx11 PROPERTY OUTPUT_NAME adios2${ADIOS2_LIBRARY_SUF
 target_link_libraries(adios2_cxx11 PRIVATE adios2_core adios2::thirdparty::pugixml)
 target_compile_features(adios2_cxx11 INTERFACE ${ADIOS2_CXX11_FEATURES})

+if(ADIOS2_HAVE_Kokkos)
+    target_link_libraries(adios2_cxx11 PUBLIC Kokkos::kokkos)
+endif()
+
 target_include_directories(adios2_cxx11
   PUBLIC
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
```

Add functions for Put and Get that take a Kokkos view

```diff
diff --git a/bindings/CXX11/adios2/cxx11/Engine.cpp b/bindings/CXX11/adios2/cxx11/Engine.cpp
index 1b140309e..6d58b8ddd 100644
--- a/bindings/CXX11/adios2/cxx11/Engine.cpp
+++ b/bindings/CXX11/adios2/cxx11/Engine.cpp
@@ -344,6 +344,18 @@ Engine::AllStepsBlocksInfo(const VariableNT &variable) const
     return ret;
 }

+#ifdef ADIOS2_HAVE_KOKKOS
+#define declare_template_instantiation(T, MemSpace)                            \
+    template void Engine::Put<T, MemSpace>(                                    \
+        Variable<T>, Kokkos::View<T *, MemSpace>, const Mode);                 \
+                                                                               \
+    template void Engine::Get<T, MemSpace>(                                    \
+        Variable<T>, Kokkos::View<T *, MemSpace>, const Mode);
+
+ADIOS2_FOREACH_KOKKOS_TYPE_2ARGS(declare_template_instantiation)
+#undef declare_template_instantiation
+#endif
+
 #define declare_template_instantiation(T)                                      \
     template void Engine::Put<T>(Variable<T>, const T *, const Mode);          \
     template void Engine::Put<T>(const std::string &, const T *, const Mode);  \
```
  
```diff
diff --git a/bindings/CXX11/adios2/cxx11/Engine.h b/bindings/CXX11/adios2/cxx11/Engine.h
index 166679c79..bd50aca24 100644
--- a/bindings/CXX11/adios2/cxx11/Engine.h
+++ b/bindings/CXX11/adios2/cxx11/Engine.h
@@ -18,6 +18,10 @@
 #include "adios2/common/ADIOSMacros.h"
 #include "adios2/common/ADIOSTypes.h"

+#ifdef ADIOS2_HAVE_KOKKOS
+#include <Kokkos_Core.hpp>
+#endif
+
 namespace adios2
 {
@@ -211,6 +215,23 @@ public:
      * collective call and can only be called between Begin/EndStep pairs. */
     void PerformDataWrite();

+#ifdef ADIOS2_HAVE_KOKKOS
+    /** Get and Put functions for Kokkos buffers */
+    template <class T, class MemSpace>
+    void Put(Variable<T> variable, Kokkos::View<T *, MemSpace> data,
+             const Mode launch = Mode::Deferred);
+
+    template <class T, class MemSpace>
+    void Get(Variable<T> variable, Kokkos::View<T *, MemSpace> data,
+             const Mode launch = Mode::Deferred);
+#endif
+
     /**
      * Get data associated with a Variable from the Engine
      * @param variable contains variable metadata information
```

```diff
diff --git a/bindings/CXX11/adios2/cxx11/Engine.tcc b/bindings/CXX11/adios2/cxx11/Engine.tcc
index e7bcebd91..734363dc7 100644
--- a/bindings/CXX11/adios2/cxx11/Engine.tcc
+++ b/bindings/CXX11/adios2/cxx11/Engine.tcc
@@ -137,6 +137,28 @@ void Engine::Put(const std::string &variableName, const T &datum,
                   launch);
 }

+#ifdef ADIOS2_HAVE_KOKKOS
+template <class T, class MemSpace>
+void Engine::Put(Variable<T> variable, Kokkos::View<T *, MemSpace> data,
+                 const Mode launch)
+{
+    using IOType = typename TypeInfo<T>::IOType;
+    adios2::helper::CheckForNullptr(m_Engine, "in call to Engine::Put");
+    adios2::helper::CheckForNullptr(variable.m_Variable,
+                                    "for variable in call to Engine::Put");
+    m_Engine->Put(*variable.m_Variable,
+                  reinterpret_cast<const IOType *>(data.data()), launch);
+}
+#endif
+
 template <class T>
 void Engine::Get(Variable<T> variable, T *data, const Mode launch)
 {
 @@ -234,6 +256,28 @@ void Engine::Get(Variable<T> variable, T **data) const
     return;
 }

+#ifdef ADIOS2_HAVE_KOKKOS
+template <class T, class MemSpace>
+void Engine::Get(Variable<T> variable, Kokkos::View<T *, MemSpace> data,
+                 const Mode launch)
+{
+    adios2::helper::CheckForNullptr(variable.m_Variable,
+                                    "for variable in call to Engine::Get");
+    using IOType = typename TypeInfo<T>::IOType;
+    adios2::helper::CheckForNullptr(m_Engine, "in call to Engine::Get");
+    m_Engine->Get(*variable.m_Variable, reinterpret_cast<IOType *>(data.data()),
+                  launch);
+}
+#endif
+
 template <class T>
 std::map<size_t, std::vector<typename Variable<T>::Info>>
 Engine::AllStepsBlocksInfo(const Variable<T> variable) const
```

Instantiate Kokkos::View Put/Get functions for all supported memory spaces

```diff
diff --git a/source/adios2/CMakeLists.txt b/source/adios2/CMakeLists.txt
index e7bcebd91..734363dc7 100644
--- a/source/adios2/CMakeLists.txt
+++ b/source/adios2/CMakeLists.txt
@@ -107,6 +107,10 @@ add_library(adios2_core
set_property(TARGET adios2_core PROPERTY EXPORT_NAME core)
set_property(TARGET adios2_core PROPERTY OUTPUT_NAME adios2${ADIOS2_LIBRARY_SUFFIX}_core)

+if(ADIOS2_HAVE_Kokkos)
+  target_link_libraries(adios2_core PUBLIC Kokkos::kokkos)
+endif()
+
set(maybe_adios2_core_cuda)
if(ADIOS2_HAVE_CUDA)
  add_library(adios2_core_cuda helper/adiosCUDA.cu)
```

```diff
diff --git a/source/adios2/common/ADIOSMacros.h b/source/adios2/common/ADIOSMacros.h
index e7bcebd91..734363dc7 100644
--- a/source/adios2/common/ADIOSMacros.h
+++ b/source/adios2/common/ADIOSMacros.h
@@ -16,6 +16,9 @@

#include "adios2/common/ADIOSTypes.h"

+#ifdef ADIOS2_HAVE_KOKKOS
+#include <Kokkos_Core.hpp>
+#endif
/**
 <pre>
 The ADIOS_FOREACH_TYPE_1ARG macro assumes the given argument is a macro which
@@ -266,4 +269,36 @@
    iterator begin() noexcept { return iterator(DATA_FUNCTION); }              \
    iterator end() noexcept { return iterator(DATA_FUNCTION + SIZE_FUNCTION); }

+#if defined(ADIOS2_HAVE_KOKKOS) && defined(KOKKOS_ENABLE_CUDA)
+#define ADIOS2_FOREACH_KOKKOS_TYPE_2ARGS(MACRO)                                \
+    MACRO(int32_t, Kokkos::CudaSpace)                                          \
+    MACRO(uint32_t, Kokkos::CudaSpace)                                         \
+    MACRO(int64_t, Kokkos::CudaSpace)                                          \
+    MACRO(uint64_t, Kokkos::CudaSpace)                                         \
+    MACRO(float, Kokkos::CudaSpace)                                            \
+    MACRO(double, Kokkos::CudaSpace)                                           \
+    MACRO(int32_t, Kokkos::CudaHostPinnedSpace)                                \
+    MACRO(uint32_t, Kokkos::CudaHostPinnedSpace)                               \
+    MACRO(int64_t, Kokkos::CudaHostPinnedSpace)                                \
+    MACRO(uint64_t, Kokkos::CudaHostPinnedSpace)                               \
+    MACRO(float, Kokkos::CudaHostPinnedSpace)                                  \
+    MACRO(double, Kokkos::CudaHostPinnedSpace)                                 \
+    MACRO(int32_t, Kokkos::CudaUVMSpace)                                       \
+    MACRO(uint32_t, Kokkos::CudaUVMSpace)                                      \
+    MACRO(int64_t, Kokkos::CudaUVMSpace)                                       \
+    MACRO(uint64_t, Kokkos::CudaUVMSpace)                                      \
+    MACRO(float, Kokkos::CudaUVMSpace)                                         \
+    MACRO(double, Kokkos::CudaUVMSpace)
+#endif
+
+#ifdef ADIOS2_HAVE_KOKKOS
+#define ADIOS2_FOREACH_KOKKOS_TYPE_2ARGS(MACRO)                                \
+    MACRO(int32_t, Kokkos::HostSpace)                                          \
+    MACRO(uint32_t, Kokkos::HostSpace)                                         \
+    MACRO(int64_t, Kokkos::HostSpace)                                          \
+    MACRO(uint64_t, Kokkos::HostSpace)                                         \
+    MACRO(float, Kokkos::HostSpace)                                            \
+    MACRO(double, Kokkos::HostSpace)
+#endif
+
#endif /* ADIOS2_ADIOSMACROS_H */
```
