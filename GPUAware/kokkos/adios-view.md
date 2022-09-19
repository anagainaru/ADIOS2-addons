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

Add the header files so users can include the KokkosView.

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
@@ -82,14 +80,12 @@ install(
 install(
   FILES adios2/cxx11/ADIOS.h
         adios2/cxx11/ADIOS.inl
+        adios2/cxx11/ADIOSView.h
         adios2/cxx11/IO.h
         adios2/cxx11/Group.h
         adios2/cxx11/Variable.h
         adios2/cxx11/VariableNT.h
         adios2/cxx11/Attribute.h
         adios2/cxx11/Engine.h
+        adios2/cxx11/KokkosView.h
         adios2/cxx11/Operator.h
         adios2/cxx11/Query.h
         adios2/cxx11/Types.h
```

ADIOS View stub class:
```diff
diff --git a/bindings/CXX11/adios2/cxx11/ADIOSView.h b/bindings/CXX11/adios2/cxx11/ADIOSView.h
new file mode 100644
index 000000000..3962651ec
--- /dev/null
+++ b/bindings/CXX11/adios2/cxx11/ADIOSView.h
@@ -0,0 +1,14 @@
+#ifndef ADIOS2_BINDINGS_CXX11_CXX11_ADIOS_VIEW_H_
+#define ADIOS2_BINDINGS_CXX11_CXX11_ADIOS_VIEW_H_
+
+namespace adios2
+{
+template <typename T, class... Parameters>
+class AdiosView
+{
+public:
+    AdiosView() = delete;
+};
+}
+
+#endif /* ADIOS2_BINDINGS_CXX11_CXX11_ADIOS_VIEW_H_ */
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
@@ -201,6 +202,27 @@ public:
     void Put(const std::string &variableName, const T &datum,
              const Mode launch = Mode::Deferred);

+    /**
+     * The next two Put functions are used to accept a variable, and an
+     * AdiosViews which is a placeholder for Kokkos::View
+     * @param variable contains variable metadata information
+     * @param data represents any user defined object that is not a vector (used
+     * for an AdiosView)
+     * @param launch mode policy, optional for API consistency, internally is
+     * always sync
+     */
+    template <class T, typename U,
+              class = typename std::enable_if<
+                  std::is_convertible<U, AdiosView<U>>::value>::type>
+    void Put(Variable<T> variable, U const &data,
+             const Mode launch = Mode::Deferred)
+    {
+        auto adios_data = static_cast<AdiosView<U>>(data);
+        auto mem_space = adios_data.memory_space();
+        variable.SetMemorySpace(mem_space);
+        Put(variable, adios_data.data(), launch);
+    }
+
@@ -391,6 +413,27 @@ public:
     template <class T>
     void Get(Variable<T> variable, T **data) const;

+    /**
+     * The next two Get functions are used to accept a variable, and an
+     * AdiosViews which is a placeholder for Kokkos::View
+     * @param variable contains variable metadata information
+     * @param data represents any user defined object that is not a vector (used
+     * for an AdiosView)
+     * @param launch mode policy, optional for API consistency, internally is
+     * always sync
+     */
+    template <class T, typename U,
+              class = typename std::enable_if<
+                  std::is_convertible<U, AdiosView<U>>::value>::type>
+    void Get(Variable<T> variable, U const &data,
+             const Mode launch = Mode::Deferred)
+    {
+        auto adios_data = static_cast<AdiosView<U>>(data);
+        auto mem_space = adios_data.memory_space();
+        variable.SetMemorySpace(mem_space);
+        Get(variable, adios_data.data(), launch);
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
+template <class T, class... Parameters>
+class AdiosView<Kokkos::View<T, Parameters...>>
+{
+    using data_type = typename Kokkos::View<T, Parameters...>::value_type;
+    data_type *pointer;
+    adios2::MemorySpace mem_space;
+
+public:
+    template <class... P>
+    AdiosView(Kokkos::View<T, P...> v)
+    {
+        pointer = v.data();
+        mem_space = detail::memspace_kokkos_to_adios2<
+            typename Kokkos::View<T, P...>::memory_space>::value;
+    }
+
+    data_type const *data() const { return pointer; }
+    data_type *data() { return pointer; }
+    adios2::MemorySpace memory_space() const { return mem_space; }
+};
+
+}
+#endif
```
