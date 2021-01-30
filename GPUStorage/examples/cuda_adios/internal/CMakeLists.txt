#------------------------------------------------------------------------------#
# Distributed under the OSI-approved Apache License, Version 2.0.  See
# accompanying file Copyright.txt for details.
#------------------------------------------------------------------------------#

if(ADIOS2_HAVE_CUDA)
  add_executable(GPUWriteRead_cuda bpWriteRead.cpp)
  #message (FATAL_ERROR "${CUDA_LIBRARIES} ${CUDA_INCLUDE_DIRS}")
  target_include_directories(GPUWriteRead_cuda PUBLIC ${CUDA_INCLUDE_DIRS})
  target_link_libraries(GPUWriteRead_cuda PUBLIC adios2::cxx11 ${CUDA_LIBRARIES})
else()
  add_executable(GPUWriteRead bpWriteRead.cpp)
  target_link_libraries(GPUWriteRead adios2::cxx11)
endif()
