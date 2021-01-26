# Changes to the ADIOS2 library 

Changes to the ADIOS library follow the diagram below

![ADIOS workflow](docs/ADIOS_GDS.png)

In order to allow the application code to call `adios2::IO::Put` using GPU buffers, the ADIOS library requires several new classes and changes to the cmake files.

## Code changes

List of files that require changes in ADIOS:
1. New transport for GPUDirect
```
    source/adios2/toolkit/transport/gpu/GPUdirect.cpp
    source/adios2/toolkit/transport/gpu/GPUdirect.h
```
2. Utility functions for the new transport
```
    source/adios2/toolkit/format/bp/BPBase.cpp
    source/adios2/toolkit/format/bp/BPBase.h
    source/adios2/toolkit/format/bp/bp4/BP4Base.cpp
    source/adios2/toolkit/format/bp/bp4/BP4Base.h
```
3. Changes in the File Engine
```
    source/adios2/engine/bp4/BP4Writer.cpp
    source/adios2/engine/bp4/BP4Writer.h
```
4. Compiling
```
   examples/CMakeLists.txt
   source/adios2/CMakeLists.txt
```

Each bullet is described bellow.

### 1. Create a new transport for GPU direct

**Files available in this repo in `adios/transport`.**

The transport is implemented in the `GPUdirect.*` files from `source/adios2/toolkit/transport/gpu/`. The code implements the open, read, write and close functions for direct access between GPU and storage (in the exact same way it's being used in the examples in this repo).

Add the new files to the libraries compiled in the `adios2_core` target (in `source/adios2/CMakeLists.txt`):
```
if(ADIOS2_HAVE_CUDA)
  set(adios2_core_sources ${adios2_core_sources}
    toolkit/transport/gpu/GPUdirect.cpp
  )
endif()

add_library(adios2_core "${adios2_core_sources}")
```

### 2. Utility functions for the new transport

**Functions for returning the name of the bp gpu files**

Changes inside `ADIOS2/source/adios2/toolkit/format/bp/bp4/BP4Base.*`.

```c++
std::vector<std::string>
BP4Base::GetBPGPUFileNames(const std::vector<std::string> &names) const
    noexcept
{
    std::vector<std::string> gpuFileNames;
    gpuFileNames.reserve(names.size());
    for (const auto &name : names)
    {
        gpuFileNames.push_back(GetBPGPUFileName(name, m_RankMPI, m_RankGPU));
    }
    return gpuFileNames;
}

std::string BP4Base::GetBPGPUFileName(const std::string &name,
    const size_t indexMPI, const size_t indexGPU) const
    noexcept
{
    const std::string bpName = helper::RemoveTrailingSlash(name);
    const std::string bpGPUDataRankName(bpName + PathSeparator + "gpu." +
                                         std::to_string(indexMPI) + "." +
                                         std::to_string(indexGPU));
    return bpGPUDataRankName;
}
```

Files written through GPU direct will be stored in `<base_name.bp>/gpu.<MPI rank>.<GPU id>`.

**Store the GPU id in a variable**

The `m_RankGPU` will be added in `source/adios2/toolkit/format/bp/BPBase.*`.

```c++
    #ifdef ADIOS2_HAVE_CUDA
        cudaGetDevice(&m_RankGPU);
    #endif
```

### 3. Changes in the File Engine

**3.1 Constructor**
```
BP4Writer::BP4Writer(IO &io, const std::string &name, const Mode mode,
                     helper::Comm comm)
: Engine("BP4Writer", io, name, mode, std::move(comm)), m_BP4Serializer(m_Comm),
  m_FileDataManager(m_Comm), m_GPUDataManager(m_Comm),
  m_FileMetadataManager(m_Comm), m_FileMetadataIndexManager(m_Comm),
  m_FileDrainer()
{
```

**3.2 Initializing the transports**

Inside the BP4Writer class in the initTransports function, the gpu files need to be opened.

```c++
        m_GPUStreamNames = m_BP4Serializer.GetBPGPUFileNames(transportsNames);
        
        ...
        
        #ifdef ADIOS2_HAVE_CUDA
            Params defaultTransportParameters;
            defaultTransportParameters["transport"] = "GPU";
            std::vector<Params> tempTransportsParameters.push_back(
                            defaultTransportParameters);
            m_GPUDataManager.OpenFiles(m_GPUStreamNames, m_OpenMode,
                                       m_IO.m_TransportsParameters,
                                       m_BP4Serializer.m_Profiler.m_IsActive);
        #endif
```

The OpenFiles function inside Transportman needs to change to allow GPU libraries.

**3.3 Detect if the buffer sent by the user is in GPU memeory space**

**3.4 Write data usung GPUdirect**


## 4. Compiling

### 4.1 Add CUDA package to cmake when ADIOS detects the CUDA compiler

Add the Cuda compiler checks in `{ADIOS_ROOT}/CMakeList.txt`:
- Add an option to have Cuda `adios_option(CUDA "Enable support for Cuda" AUTO)`
- Add Cuda when setting the Config options in `ADIOS2_CONFIG_OPTS`
- Print information about the Cuda compiler

```
if(ADIOS2_HAVE_CUDA)
       message("  Cuda Compiler : ${CMAKE_CUDA_COMPILER} ")
endif()
```

If the Cuda compiler is found, find the cmake package for it in `{ADIOS_ROOT}/cmake/DetectOptions.cmake`
```
if(ADIOS2_USE_CUDA STREQUAL AUTO)
  find_package(CUDA)
elseif(ADIOS2_USE_CUDA)
  find_package(CUDA REQUIRED)
endif()
if(CUDA_FOUND)
  set(ADIOS2_HAVE_CUDA TRUE)
endif()
```

### 4.2. Compile ADIOS with CUDA and CuFile enabled

**File available in this repo in `adios/compile`.**

Add the libraries and include directories for Cuda and CuFile in `{ADIOS_ROOT}/source/adios2/CmakeLists.txt`.

For CUDA the link to the Cuda compiler needs to be added:
```
if(ADIOS2_HAVE_CUDA)
  target_include_directories(adios2_core PUBLIC ${CUDA_INCLUDE_DIRS})
  target_link_libraries(adios2_core PUBLIC ${CUDA_LIBRARIES})
endif()
```

For CuFile, we manually add the libraries required by GDS (installed by following the steps [here](./README.md):
```
if(ADIOS2_HAVE_CUDA)
   target_include_directories(adios2_core PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda-11.1/targets/x86_64-linux/lib/)
   target_link_directories(adios2_core PUBLIC /usr/local/cuda-11.1/targets/x86_64-linux/l
ib/)
   target_link_libraries(adios2_core PUBLIC ${CUDA_LIBRARIES} -lcufile)
endif()
```

