## ADIOS2 reorganize script using GPU

Changes to the code implemented in `source/utils/adios_reorganize/Reorganize.cpp` to read and write on the GPU.

This is useful to reproduce performance results when using GPU pointers for a given dataset.

### Usage

The changes need to be applied to the util files inside the adios2 library.
```
$ ./bin/adios2_reorganize
Usage: adios_reorganize input output rmethod "params" wmethod "params" <decomposition>
    input   Input stream path
    output  Output file path
    rmethod ADIOS method to read with
            Supported read methods: BPFile, HDF5, SST, SSC, DataMan
    params  Read method parameters (in quotes; comma-separated list)
    wmethod ADIOS method to write with
    params  Write method parameters (in quotes; comma-separated list)
    <decomposition>    list of numbers e.g. 32 8 4
            Decomposition values in each dimension of an array
            The product of these number must be less then the number
            of processes. Processes whose rank is higher than the
            product, will not write anything.
               Arrays with less dimensions than the number of values,
            will be decomposed with using the appropriate number of
            values.
$ ./bin/adios2_reorganize BP4Test.bp BP5Test.bp BP4 "" BP5 ""
Input stream            = BP4Test.bp
Output stream           = BP5Test.bp
Read method             = BP4
Read method parameters  =
Write method            = BP5
Write method parameters =
Waiting to open stream BP4Test.bp...
____________________

File info:
  current step:   0
  # of variables: 2
  # of attributes: 0
Get info on variable 0: bpStep
    uint32_t bpStep	scalar
Get info on variable 1: bpFloats
    float bpFloats[60000]
rank 0: position in 1-D decomposition = 0
rank 0: ldims in 1-D space = {60000}
rank 0: offsets in 1-D space = {0}
rank 0: Read variable bpStep
rank 0: Read variable bpFloats
rank 0: Write variable bpStep
rank 0: Write variable bpFloats
```

### Code changes
```diff
diff --git a/source/utils/CMakeLists.txt b/source/utils/CMakeLists.txt
index 30dd48411..0ca4ab041 100644
--- a/source/utils/CMakeLists.txt
+++ b/source/utils/CMakeLists.txt
@@ -59,13 +59,18 @@ if(ADIOS2_HAVE_MGARD)
 else()
   set(maybe_mgard)
 endif()
+if(ADIOS2_HAVE_Kokkos)
+  set(maybe_kokkos Kokkos::kokkos)
+else()
+  set(maybe_kokkos)
+endif()
 add_executable(adios_reorganize ${adios_reorganize_srcs})
-target_link_libraries(adios_reorganize PRIVATE adios2_core ${maybe_mgard})
+target_link_libraries(adios_reorganize PRIVATE adios2_core ${maybe_mgard} ${maybe_kokkos})
 set_property(TARGET adios_reorganize PROPERTY OUTPUT_NAME adios2_reorganize${ADIOS2_EXECUTABLE_SUFFIX})

 if(ADIOS2_HAVE_MPI)
   add_executable(adios_reorganize_mpi ${adios_reorganize_srcs})
-  target_link_libraries(adios_reorganize_mpi PRIVATE adios2_core_mpi ${maybe_mgard})
+  target_link_libraries(adios_reorganize_mpi PRIVATE adios2_core_mpi ${maybe_mgard} ${maybe_kokkos})
   set_property(TARGET adios_reorganize_mpi PROPERTY OUTPUT_NAME adios2_reorganize_mpi${ADIOS2_EXECUTABLE_SUFFIX})
   set(maybe_adios_reorganize_mpi adios_reorganize_mpi)
 else()
diff --git a/source/utils/adios_reorganize/Reorganize.cpp b/source/utils/adios_reorganize/Reorganize.cpp
index dc1288fe1..e2221ed62 100644
--- a/source/utils/adios_reorganize/Reorganize.cpp
+++ b/source/utils/adios_reorganize/Reorganize.cpp
@@ -42,6 +42,8 @@
 #include <cerrno>
 #include <cstdlib>

+#include <Kokkos_Core.hpp>
+
 namespace adios2
 {
 namespace utils
@@ -320,13 +322,6 @@ std::vector<VarInfo> varinfo;
 //
 void Reorganize::CleanUpStep(core::IO &io)
 {
-    for (auto &vi : varinfo)
-    {
-        if (vi.readbuf != nullptr)
-        {
-            free(vi.readbuf);
-        }
-    }
     varinfo.clear();
     // io.RemoveAllVariables();
     // io.RemoveAllAttributes();
@@ -609,14 +604,15 @@ int Reorganize::ReadWrite(core::Engine &rStream, core::Engine &wStream, core::IO
     }

     /*
-     * Read all variables into memory
+     * Read each variable into a Kokkos::View
      */
+    std::vector<Kokkos::View<char *>> readbuf;
+    readbuf.reserve(nvars);
     for (size_t varidx = 0; varidx < nvars; ++varidx)
     {
         if (varinfo[varidx].v != nullptr)
         {
             const std::string &name = varinfo[varidx].v->m_Name;
-            assert(varinfo[varidx].readbuf == nullptr);
             if (varinfo[varidx].writesize != 0)
             {
                 // read variable subset
@@ -625,20 +621,22 @@ int Reorganize::ReadWrite(core::Engine &rStream, core::Engine &wStream, core::IO
                 if (type == DataType::Struct)
                 {
                     // not supported
+                    std::cout << "Struct datatype are not supported for variable: " << name << std::endl;
                 }
 #define declare_template_instantiation(T)                                                          \
     else if (type == helper::GetDataType<T>())                                                     \
     {                                                                                              \
-        varinfo[varidx].readbuf = calloc(1, varinfo[varidx].writesize);                            \
+        Kokkos::View<char *> view_buf(name, varinfo[varidx].writesize); \
+        readbuf.push_back(view_buf); \
         if (varinfo[varidx].count.size() == 0)                                                     \
         {                                                                                          \
-            rStream.Get<T>(name, reinterpret_cast<T *>(varinfo[varidx].readbuf),                   \
+            rStream.Get<T>(name, reinterpret_cast<T *>(view_buf.data()),                   \
                            adios2::Mode::Sync);                                                    \
         }                                                                                          \
         else                                                                                       \
         {                                                                                          \
             varinfo[varidx].v->SetSelection({varinfo[varidx].start, varinfo[varidx].count});       \
-            rStream.Get<T>(name, reinterpret_cast<T *>(varinfo[varidx].readbuf));                  \
+            rStream.Get<T>(name, reinterpret_cast<T *>(view_buf.data()));                  \
         }                                                                                          \
     }
                 ADIOS2_FOREACH_STDTYPE_1ARG(declare_template_instantiation)
@@ -671,18 +669,18 @@ int Reorganize::ReadWrite(core::Engine &rStream, core::Engine &wStream, core::IO
     {                                                                                              \
         if (varinfo[varidx].count.size() == 0)                                                     \
         {                                                                                          \
-            wStream.Put<T>(name, reinterpret_cast<T *>(varinfo[varidx].readbuf),                   \
+            wStream.Put<T>(name, reinterpret_cast<T *>(readbuf[varidx].data()),                   \
                            adios2::Mode::Sync);                                                    \
         }                                                                                          \
         else if (varinfo[varidx].v->m_ShapeID == adios2::ShapeID::LocalArray)                      \
         {                                                                                          \
-            wStream.Put<T>(name, reinterpret_cast<T *>(varinfo[varidx].readbuf),                   \
+            wStream.Put<T>(name, reinterpret_cast<T *>(readbuf[varidx].data()),                   \
                            adios2::Mode::Sync);                                                    \
         }                                                                                          \
         else                                                                                       \
         {                                                                                          \
             varinfo[varidx].v->SetSelection({varinfo[varidx].start, varinfo[varidx].count});       \
-            wStream.Put<T>(name, reinterpret_cast<T *>(varinfo[varidx].readbuf));                  \
+            wStream.Put<T>(name, reinterpret_cast<T *>(readbuf[varidx].data()));                  \
         }                                                                                          \
     }
                 ADIOS2_FOREACH_STDTYPE_1ARG(declare_template_instantiation)
diff --git a/source/utils/adios_reorganize/Reorganize.h b/source/utils/adios_reorganize/Reorganize.h
index 4c8ffab6f..55969d2df 100644
--- a/source/utils/adios_reorganize/Reorganize.h
+++ b/source/utils/adios_reorganize/Reorganize.h
@@ -15,6 +15,8 @@
 #include "adios2/helper/adiosComm.h"
 #include "utils/Utils.h"

+#include <vector>
+
 namespace adios2
 {
 namespace utils
@@ -27,7 +29,6 @@ struct VarInfo
     Dims start;
     Dims count;
     size_t writesize = 0;    // size of subset this process writes, 0: do not write
-    void *readbuf = nullptr; // read in buffer
 };

 class Reorganize : public Utils
```
