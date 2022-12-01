# ADIOS2 with SCR

## API

In order for ADIOS to run with SCR, it needs to be build with SCR support.
```
cmake \
  -DADIOS2_USE_SCR=ON \
  -DADIOS2_USE_MPI=ON \
  -DSCR_ROOT=/path/to/scr-v3.0.1/install \
  ../ADIOS2
```

By default SCR is not used for data transfers even if ADIOS2 is build with SCR. The `UseSCR` parameter needs to be selected in order for SCR to do anything.
If the `UseSCR` flag is set, the SCR library needs to be initialized and finalized in the user code.

```c++
#include "scr.h"

MPI_Init();
SCR_Init();

adios2::IO io = adios.DeclareIO("TestIO");
io.SetParameters({{"UseSCR", "1"}});

adios2::Engine engine = io.Open(fname, adios2::Mode::Write);

for (step=0; step<total_steps; step++)
{
  engine.Put(var, data);
}

engine.Close();

SCR_Finalize();
MPI_Finalize();
```

SCR works only for MPI codes. The library needs to be initialized after `MPI_Init` and finalized before `MPI_Finalize`.
If the flag is set without SCR being initialized an error will occur.

```
SCR v3.0.0 ABORT: rank -2 on (null): Must call SCR_Init() before SCR_Start_output() @ /ccs/home/againaru/adios/ADIOS2-scr/scr-v3.0.1/scr/src/scr.c:3161
```

If SCR is initialized before MPI was initialized an error will occur.
```
*** The MPI_Comm_dup() function was called before MPI_INIT was invoked.
```

## Code changes in ADIOS2

In `source/adios2/common/ADIOSTypes.h`

```c++
using Params = std::map<std::string, std::string>;

```

In `source/adios2/core/IO.cpp`
```c++
Params m_Parameters;
    
void IO::SetParameters(const Params &parameters) noexcept
{
    for (const auto &parameter : parameters)
    {
        m_Parameters[parameter.first] = parameter.second;
    }
}
```

Define the flag in `source/adios2/toolkit/format/bp/BPBase.h` and set it in `source/adios2/toolkit/format/bp/BPBase.cpp`.
```c++
    /** groups all user-level parameters in a single struct */
    struct Parameters
    {
        /* Use the Scalable Checkpoint Restart (SCR) library for draining the Burst Buffers*/
        bool UseSCR = false;
    }
```
