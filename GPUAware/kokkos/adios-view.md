# Enable Get/Put with Kokkos views

Without making ADIOS dependent on Kokkos.

Create an ADIOS View stub that will be used as a place holder for Kokkos Views and use the stub in internally in the library. The user will have to include the ADIOSKokkos header.


<img width="752" alt="Screen Shot 2022-09-19 at 6 40 40 PM" src="https://user-images.githubusercontent.com/16229479/191131922-8351b889-bb4b-4a60-8e3a-7bb234ec7c4b.png">

Code changes include the following steps:
  - Create a stub ADIOS View declaring the object that will contain the Kokkos View and will be responsible for all Kokkos related functions
  - Create the ADIOS View definition linking to the Kokkos library (to be included separately by the application)
  - Create a mechanism for Kokkos to be build with Kokkos (to be able to run examples within the library)

## Interface

```c++
#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>
#include <Kokkos_Core.hpp>

...
    Kokkos::View<float*, Kokkos::HostSpace> gpuSimData("simBuffer", N);
    auto data = io.DefineVariable<float>("data", shape, start, count);
    for (steps){
        bpWriter.BeginStep();
        bpWriter.Put(data, gpuSimData);
        bpWriter.EndStep();
    }
```

## ADIOS View

```diff
diff --git a/bindings/CXX11/CMakeLists.txt b/bindings/CXX11/CMakeLists.txt
index 7815a5f33..d9f6231f2 100644
--- a/bindings/CXX11/CMakeLists.txt
+++ b/bindings/CXX11/CMakeLists.txt
@@ -5,6 +5,7 @@

 add_library(adios2_cxx11
   adios2/cxx11/ADIOS.cpp
+  adios2/cxx11/AdiosView.cpp
   adios2/cxx11/Attribute.cpp
   adios2/cxx11/Engine.cpp
   adios2/cxx11/Engine.tcc
diff --git a/bindings/CXX11/adios2/cxx11/ADIOSView.cpp b/bindings/CXX11/adios2/cxx11/ADIOSView.cpp
new file mode 100644
index 000000000..07ba1340d
--- /dev/null
+++ b/bindings/CXX11/adios2/cxx11/ADIOSView.cpp
@@ -0,0 +1,16 @@
+#ifndef ADIOS2_BINDINGS_CXX11_CXX11_ADIOSVIEW_CPP_
+#define ADIOS2_BINDINGS_CXX11_CXX11_ADIOSVIEW_CPP_
+
+#include "ADIOSView.h"
+
+namespace adios2
+{
+template <class T>
+class AdiosView{
+    public:
+    virtual void data() const = 0;
+    virtual void memory_space() const = 0;
+};
+}
+
+#endif /* ADIOS2_BINDINGS_CXX11_CXX11_ADIOSVIEW_CPP_ */
diff --git a/bindings/CXX11/adios2/cxx11/ADIOSView.h b/bindings/CXX11/adios2/cxx11/ADIOSView.h
new file mode 100644
index 000000000..f3e9c4f03
--- /dev/null
+++ b/bindings/CXX11/adios2/cxx11/ADIOSView.h
@@ -0,0 +1,9 @@
+#ifndef ADIOS2_BINDINGS_CXX11_CXX11_ADIOSVIEW_H_
+#define ADIOS2_BINDINGS_CXX11_CXX11_ADIOSVIEW_H_
+
+namespace adios2
+{
+template <class T> class AdiosView;
+}
+
+#endif /* ADIOS2_BINDINGS_CXX11_CXX11_ADIOSVIEW_H_ */
```

Used in the Engine object.

```diff
diff --git a/bindings/CXX11/adios2/cxx11/Engine.h b/bindings/CXX11/adios2/cxx11/Engine.h
index 166679c79..72171ed84 100644
--- a/bindings/CXX11/adios2/cxx11/Engine.h
+++ b/bindings/CXX11/adios2/cxx11/Engine.h
@@ -11,6 +11,7 @@
 #ifndef ADIOS2_BINDINGS_CXX11_CXX11_ENGINE_H_
 #define ADIOS2_BINDINGS_CXX11_CXX11_ENGINE_H_

+#include "ADIOSView.h"
 #include "Types.h"
 #include "Variable.h"
 #include "VariableNT.h"
@@ -201,6 +202,14 @@ public:
     void Put(const std::string &variableName, const T &datum,
              const Mode launch = Mode::Deferred);

+    /* Function using the place holder for Kokkos::View */
+    template <class T>
+    void Put(Variable<T> variable, AdiosView<T> const &data,
+             const Mode launch = Mode::Deferred)
+    {
+        Put(variable, data.data(), launch);
+    }
+
     /** Perform all Put calls in Deferred mode up to this point.  Specifically,
      * this causes Deferred data to be copied into ADIOS internal buffers as if
      * the Put had been done in Sync mode. */
@@ -391,6 +400,17 @@ public:
     template <class T>
     void Get(Variable<T> variable, T **data) const;

+    /* Function using the place holder for Kokkos::View */
+    template <class T, class U>
+    void Get(Variable<T> variable, AdiosView<U> data,
+             const Mode launch = Mode::Deferred)
+    {
+        static_assert(std::is_same<T, U>::value, "The Variable type and the View type must match");
+        auto mem_space = data.memory_space();
+        variable.SetMemorySpace(mem_space);
+        Get(variable, data.data(), launch);
+    }
+
     /** Perform all Get calls in Deferred mode up to this point */
     void PerformGets();
```

## Kokkos View

```diff
diff --git a/bindings/CXX11/adios2/cxx11/KokkosView.h b/bindings/CXX11/adios2/cxx11/KokkosView.h
new file mode 100644
index 000000000..599910055
--- /dev/null
+++ b/bindings/CXX11/adios2/cxx11/KokkosView.h
@@ -0,0 +1,50 @@
+#ifndef ADIOS2_BINDINGS_CXX11_CXX11_ADIOS_KOKKOS_H_
+#define ADIOS2_BINDINGS_CXX11_CXX11_ADIOS_KOKKOS_H_
+
+#include <Kokkos_Core.hpp>
+
+namespace adios2
+{
+    namespace detail
+{
+
+template <typename T>
+struct memspace_kokkos_to_adios2;
+
+template <>
+struct memspace_kokkos_to_adios2<Kokkos::HostSpace>
+{
+    static constexpr adios2::MemorySpace value = adios2::MemorySpace::Host;
+};
+
+#ifdef KOKKOS_ENABLE_CUDA
+
+template <>
+struct memspace_kokkos_to_adios2<Kokkos::CudaSpace>
+{
+    static constexpr adios2::MemorySpace value = adios2::MemorySpace::CUDA;
+};
+
+#endif
+
+} // namespace detail
+
+template <class T>
+class AdiosView
+{
+    T *pointer;
+    adios2::MemorySpace mem_space;
+public:
+    template <class D, class... P>
+    AdiosView(Kokkos::View<D, P...> v) {
+        pointer = v.data();
+        mem_space = detail::memspace_kokkos_to_adios2<typename Kokkos::View<D, P...>::memory_space>::value;
+    }
+
+    T *data() {return pointer;}
+    T *data() const {return pointer;}
+    adios2::MemorySpace memory_space(){return mem_space;}
+};
+
+}
+#endif
```
