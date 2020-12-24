# Changes to the ADIOS2 library 

In order to allow code to call `adios2::IO::Put` using GPU buffers, the ADIOS library requires a few modifications.

**Compile ADIOS with Cuda enabled**

Add the Cuda compiler checks in `CMakeList.txt` and `DetectOptions.cmake`
- Add an option to have Cuda `adios_option(CUDA "Enable support for Cuda" AUTO)`
- Add Cuda when setting the Config options in `ADIOS2_CONFIG_OPTS`
- Print information about the Cuda compiler

```
if(ADIOS2_HAVE_CUDA)
       message("  Cuda Compiler : ${CMAKE_CUDA_COMPILER} ")
endif()
```

If the Cuda compiler is found, find the cmake package for it in `cmake/DetectOptions.cmake`
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

**Check for GPU buffers inside ADIOS**

If Cuda is enabled, the ADIOS library will check if the buffer provided by the user is in GPU or CPU space. This can be done in the `Put` function implemented in `source/adios2/core/Engine.tcc` or when the buffered is copied to the adios buffer.

In the corresponding `CmakeLists.txt` file the link to the Cuda compiler needs to be added
```
if(ADIOS2_HAVE_CUDA)
  target_include_directories(adios2_core PUBLIC ${CUDA_INCLUDE_DIRS})
  target_link_libraries(adios2_core PUBLIC ${CUDA_LIBRARIES})
endif()
```


