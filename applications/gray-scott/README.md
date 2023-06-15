# Gray-Scott reaction diffusion model

Changes to the code in the ADIOS2-Examples repo.

The gray-scott simulation is using `Kokkos::View<double *>` instead of `std::vector<double>`.

```diff
diff --git a/source/cpp/gray-scott/simulation/gray-scott.h b/source/cpp/gray-scott/simulation/gray-scott.h
index 869d244..2fba18d 100644
--- a/source/cpp/gray-scott/simulation/gray-scott.h
+++ b/source/cpp/gray-scott/simulation/gray-scott.h
@@ -25,10 +26,10 @@ public:

     void init();
     void iterate();
-    void restart(std::vector<double> &u, std::vector<double> &v);
+    void restart(Kokkos::View<double *> &u, Kokkos::View<double *> &v);

-    const std::vector<double> &u_ghost() const;
-    const std::vector<double> &v_ghost() const;
+    const Kokkos::View<double *> &u_ghost() const;
+    const Kokkos::View<double *> &v_ghost() const;

     std::vector<double> u_noghost() const;
     std::vector<double> v_noghost() const;
@@ -39,7 +40,8 @@ public:
 protected:
     Settings settings;

-    std::vector<double> u, v, u2, v2;
+    using mem_space = Kokkos::DefaultExecutionSpace::memory_space;
+    Kokkos::View<double *, mem_space> u, v, u2, v2;

     int rank, procs;
     int west, east, up, down, north, south;
@@ -61,15 +63,14 @@ protected:
     void init_field();

     // Progess simulation for one timestep
-    void calc(const std::vector<double> &u, const std::vector<double> &v,
-              std::vector<double> &u2, std::vector<double> &v2);
+    void calc();
     // Compute reaction term for U
-    double calcU(double tu, double tv) const;
+    KOKKOS_FUNCTION double calcU(double tu, double tv) const;
     // Compute reaction term for V
-    double calcV(double tu, double tv) const;
+    KOKKOS_FUNCTION double calcV(double tu, double tv) const;
     // Compute laplacian of field s at (ix, iy, iz)
-    double laplacian(int ix, int iy, int iz,
-                     const std::vector<double> &s) const;
+    KOKKOS_FUNCTION double laplacian(int ix, int iy, int iz,
+                                     const Kokkos::View<double *> &s) const;

     // Exchange faces with neighbors
     void exchange(std::vector<double> &u, std::vector<double> &v) const;
@@ -81,10 +82,10 @@ protected:
     void exchange_yz(std::vector<double> &local_data) const;

     // Return a copy of data with ghosts removed
-    std::vector<double> data_noghost(const std::vector<double> &data) const;
+    std::vector<double> data_noghost(const Kokkos::View<double *> &data) const;

     // pointer version
-    void data_noghost(const std::vector<double> &data, double *no_ghost) const;
+    void data_noghost(const Kokkos::View<double *> &data, double *no_ghost) const;

     // Check if point is included in my subdomain
     inline bool is_inside(int x, int y, int z) const
@@ -105,7 +106,7 @@ protected:
         return true;
     }
     // Convert global coordinate to local index
-    inline int g2i(int gx, int gy, int gz) const
+    KOKKOS_FUNCTION int g2i(int gx, int gy, int gz) const
     {
         int x = gx - offset_x;
         int y = gy - offset_y;
 @@ -114,13 +115,13 @@ protected:
         return l2i(x + 1, y + 1, z + 1);
     }
     // Convert local coordinate to local index
-    inline int l2i(int x, int y, int z) const
+    KOKKOS_FUNCTION int l2i(int x, int y, int z) const
     {
         return x + y * (size_x + 2) + z * (size_x + 2) * (size_y + 2);
     }

 private:
-    void data_no_ghost_common(const std::vector<double> &data,
+    void data_no_ghost_common(const Kokkos::View<double *> &data,
                               double *data_no_ghost) const;
 };
```

With the implementation updated to use Kokkos calls.
```diff
diff --git a/source/cpp/gray-scott/simulation/gray-scott.cpp b/source/cpp/gray-scott/simulation/gray-scott.cpp
index a6f835f..5c59416 100644
--- a/source/cpp/gray-scott/simulation/gray-scott.cpp
+++ b/source/cpp/gray-scott/simulation/gray-scott.cpp
@@ -25,14 +25,21 @@ void GrayScott::init()

 void GrayScott::iterate()
 {
-    exchange(u, v);
-    calc(u, v, u2, v2);
-
-    u.swap(u2);
-    v.swap(v2);
+    auto temp_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);
+    auto temp_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
+    std::vector<double> vu(temp_u.data(), temp_u.data() + temp_u.size());
+    std::vector<double> vv(temp_v.data(), temp_v.data() + temp_v.size());
+    exchange(vu, vv);
+    Kokkos::deep_copy(u, temp_u);
+    Kokkos::deep_copy(v, temp_v);
+
+    calc();
+
+    std::swap(u, u2);
+    std::swap(v, v2);
 }

-void GrayScott::restart(std::vector<double> &u_in, std::vector<double> &v_in)
+void GrayScott::restart(Kokkos::View<double *> &u_in, Kokkos::View<double *> &v_in)
 {
     auto expected_len = (size_x + 2) * (size_y + 2) * (size_z + 2);
     if (u_in.size() == expected_len)
@@ -49,13 +56,19 @@ void GrayScott::restart(std::vector<double> &u_in, std::vector<double> &v_in)
     }
 }

-const std::vector<double> &GrayScott::u_ghost() const { return u; }
+const Kokkos::View<double *> &GrayScott::u_ghost() const { return u; }

-const std::vector<double> &GrayScott::v_ghost() const { return v; }
+const Kokkos::View<double *> &GrayScott::v_ghost() const { return v; }

-std::vector<double> GrayScott::u_noghost() const { return data_noghost(u); }
+std::vector<double> GrayScott::u_noghost() const
+{
+    return data_noghost(u);
+}

-std::vector<double> GrayScott::v_noghost() const { return data_noghost(v); }
+std::vector<double> GrayScott::v_noghost() const
+{
+    return data_noghost(v);
+}

 void GrayScott::u_noghost(double *u_no_ghost) const
 {
 @@ -68,14 +81,14 @@ void GrayScott::v_noghost(double *v_no_ghost) const
 }

 std::vector<double>
-GrayScott::data_noghost(const std::vector<double> &data) const
+GrayScott::data_noghost(const Kokkos::View<double *> &data) const
 {
     std::vector<double> buf(size_x * size_y * size_z);
     data_no_ghost_common(data, buf.data());
     return buf;
 }

-void GrayScott::data_noghost(const std::vector<double> &data,
+void GrayScott::data_noghost(const Kokkos::View<double *> &data,
                              double *data_no_ghost) const
 {
     data_no_ghost_common(data, data_no_ghost);
 @@ -84,13 +97,20 @@ void GrayScott::data_noghost(const std::vector<double> &data,
 void GrayScott::init_field()
 {
     const int V = (size_x + 2) * (size_y + 2) * (size_z + 2);
-    u.resize(V, 1.0);
-    v.resize(V, 0.0);
-    u2.resize(V, 0.0);
-    v2.resize(V, 0.0);
+    Kokkos::resize(u, V);
+    Kokkos::deep_copy(u, 1.0);
+    Kokkos::resize(v, V);
+    Kokkos::deep_copy(v, 0.0);
+    Kokkos::resize(u2, V);
+    Kokkos::deep_copy(u2, 0.0);
+    Kokkos::resize(v2, V);
+    Kokkos::deep_copy(v2, 0.0);

     const int d = 6;
-    for (int z = settings.L / 2 - d; z < settings.L / 2 + d; z++)
+    Kokkos::parallel_for(
+            "init_buffers",
+            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(settings.L / 2 - d, settings.L / 2 + d),
+            KOKKOS_LAMBDA(int z)
     {
         for (int y = settings.L / 2 - d; y < settings.L / 2 + d; y++)
         {
@@ -99,25 +119,25 @@ void GrayScott::init_field()
                 if (!is_inside(x, y, z))
                     continue;
                 int i = g2i(x, y, z);
-                u[i] = 0.25;
-                v[i] = 0.33;
+                u(i) = 0.25;
+                v(i) = 0.33;
             }
         }
-    }
+    });
 }

-double GrayScott::calcU(double tu, double tv) const
+KOKKOS_FUNCTION double GrayScott::calcU(double tu, double tv) const
 {
     return -tu * tv * tv + settings.F * (1.0 - tu);
 }

-double GrayScott::calcV(double tu, double tv) const
+KOKKOS_FUNCTION double GrayScott::calcV(double tu, double tv) const
 {
     return tu * tv * tv - (settings.F + settings.k) * tv;
 }

-double GrayScott::laplacian(int x, int y, int z,
-                            const std::vector<double> &s) const
+KOKKOS_FUNCTION double GrayScott::laplacian(int x, int y, int z,
+                            const Kokkos::View<double *> &s) const
 {
     double ts = 0.0;
     ts += s[l2i(x - 1, y, z)];
@@ -131,10 +151,12 @@ double GrayScott::laplacian(int x, int y, int z,
     return ts / 6.0;
 }

-void GrayScott::calc(const std::vector<double> &u, const std::vector<double> &v,
-                     std::vector<double> &u2, std::vector<double> &v2)
+void GrayScott::calc()
 {
-    for (int z = 1; z < size_z + 1; z++)
+    Kokkos::parallel_for(
+            "calc_gray_scott",
+            Kokkos::RangePolicy<>(1, size_z + 1),
+            KOKKOS_LAMBDA(int z)
     {
         for (int y = 1; y < size_y + 1; y++)
         {
@@ -145,14 +167,14 @@ void GrayScott::calc(const std::vector<double> &u, const std::vector<double> &v,
                 double dv = 0.0;
                 du = settings.Du * laplacian(x, y, z, u);
                 dv = settings.Dv * laplacian(x, y, z, v);
-                du += calcU(u[i], v[i]);
-                dv += calcV(u[i], v[i]);
-                du += settings.noise * uniform_dist(mt_gen);
-                u2[i] = u[i] + du * settings.dt;
-                v2[i] = v[i] + dv * settings.dt;
+                du += calcU(u(i), v(i));
+                dv += calcV(u(i), v(i));
+                //du += settings.noise * uniform_dist(mt_gen);
+                u2(i) = u(i) + du * settings.dt;
+                v2(i) = v(i) + dv * settings.dt;
             }
         }
-    }
+    });
 }
 @@ -268,18 +290,21 @@ void GrayScott::exchange(std::vector<double> &u, std::vector<double> &v) const
     exchange_yz(v);
 }

-void GrayScott::data_no_ghost_common(const std::vector<double> &data,
+void GrayScott::data_no_ghost_common(const Kokkos::View<double *> &data,
                                      double *data_no_ghost) const
 {
-    for (int z = 1; z < size_z + 1; z++)
-    {
+    Kokkos::parallel_for(
+            "updateBuffer",
+            Kokkos::RangePolicy<>(1, size_z + 1),
+            KOKKOS_LAMBDA(int z)
+    {
         for (int y = 1; y < size_y + 1; y++)
         {
             for (int x = 1; x < size_x + 1; x++)
             {
                 data_no_ghost[(x - 1) + (y - 1) * size_x +
-                              (z - 1) * size_x * size_y] = data[l2i(x, y, z)];
+                              (z - 1) * size_x * size_y] = data(l2i(x, y, z));
             }
         }
-    }
+    });
 }
 
```

Update the reader and writer for checkpointing and restart to use the new containers.

```diff
diff --git a/source/cpp/gray-scott/simulation/main.cpp b/source/cpp/gray-scott/simulation/main.cpp
index 6378c2d..922f098 100644
--- a/source/cpp/gray-scott/simulation/main.cpp
+++ b/source/cpp/gray-scott/simulation/main.cpp
@@ -6,10 +6,12 @@
 #include <adios2.h>
 #include <mpi.h>
 +#include <Kokkos_Core.hpp>
 @@ -81,94 +83,97 @@ int main(int argc, char **argv)
         MPI_Abort(MPI_COMM_WORLD, -1);
     }

-    Settings settings = Settings::from_json(argv[1]);
+    Kokkos::initialize(argc, argv);
+    {
 @@ -81,94 +83,97 @@ int main(int argc, char **argv)
        writer_main.close();
     }
+    Kokkos::finalize();
 #ifdef ENABLE_TIMERS
 diff --git a/source/cpp/gray-scott/simulation/restart.cpp b/source/cpp/gray-scott/simulation/restart.cpp
index 83ae858..5538ff1 100644
--- a/source/cpp/gray-scott/simulation/restart.cpp
+++ b/source/cpp/gray-scott/simulation/restart.cpp
@@ -73,13 +73,13 @@ int ReadRestart(MPI_Comm comm, const Settings &settings, GrayScott &sim,
         size_t Y = sim.size_y + 2;
         size_t Z = sim.size_z + 2;
         size_t R = static_cast<size_t>(rank);
-        std::vector<double> u, v;
+        Kokkos::View<double *> u("u", X * Y * Z), v("v", X * Y * Z);

         var_u.SetSelection({{R, 0, 0, 0}, {1, X, Y, Z}});
         var_v.SetSelection({{R, 0, 0, 0}, {1, X, Y, Z}});
         reader.Get<int>(var_step, step);
-        reader.Get<double>(var_u, u);
-        reader.Get<double>(var_v, v);
+        reader.Get<double>(var_u, u.data());
+        reader.Get<double>(var_v, v.data());
         reader.Close();

         if (!rank)
 diff --git a/source/cpp/gray-scott/simulation/writer.cpp b/source/cpp/gray-scott/simulation/writer.cpp
index f549f70..227cf7a 100644
--- a/source/cpp/gray-scott/simulation/writer.cpp
+++ b/source/cpp/gray-scott/simulation/writer.cpp
@@ -111,8 +111,8 @@ void Writer::write(int step, const GrayScott &sim)

     if (settings.adios_memory_selection)
     {
-        const std::vector<double> &u = sim.u_ghost();
-        const std::vector<double> &v = sim.v_ghost();
+        const Kokkos::View<double *> &u = sim.u_ghost();
+        const Kokkos::View<double *> &v = sim.v_ghost();

         writer.BeginStep();
         writer.Put<int>(var_step, &step);
```
