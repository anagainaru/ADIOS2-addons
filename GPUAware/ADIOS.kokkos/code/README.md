## Code using ADIOS with Kokkos::View

Running the example:
```
cmake -DKokkos_ROOT=/path/to/kokkos/install -Dadios2_ROOT=/path/to/ADIOS2/install ..
make -j
./KokkosBP4WriteRead
Steps expected by the reader: 10
Expecting data per step: 6000 elements
Simualation step 0 : 6000 elements: 0
Simualation step 1 : 6000 elements: 5
Simualation step 2 : 6000 elements: 10
Simualation step 3 : 6000 elements: 15
Simualation step 4 : 6000 elements: 20
Simualation step 5 : 6000 elements: 25
Simualation step 6 : 6000 elements: 30
Simualation step 7 : 6000 elements: 35
Simualation step 8 : 6000 elements: 40
Simualation step 9 : 6000 elements: 45
```

**Details**

- ADIOS2 and Kokkos are installed normally
- The code needs to include Kokkos and ADIOS2 header files

```c++
#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>
#include <Kokkos_Core.hpp>
```

- `Variable`s and `View`s are created normally 

```c++
    Kokkos::View<float*, Kokkos::HostSpace> gpuSimData("simBuffer", N);
    auto data = io.DefineVariable<float>("data", shape, start, count);
```

- `Put`/`Get` functions used as normal

```c++
        bpWriter.Put(data, gpuSimData);
        bpWriter.Get(data, gpuSimData);
```

**Note**

The Variable and KokkosView need to have the same type
