# OpenPMD benchmark

Instructions on running the code from [https://openpmd-api.readthedocs.io/en/0.13.3/usage/benchmarks.html](https://openpmd-api.readthedocs.io/en/0.13.3/usage/benchmarks.html)

## Writing
Code in: `/gpfs/alpine/csc143/proj-shared/againaru/warpx/openPMD-api/examples/8a_benchmark_write_parallel.cpp`

### Code

```c++
Series series = Series(filename, Access::CREATE, MPI_COMM_WORLD);
series.setMeshesPath( "fields" );
store(series, step); // includes storeMesh and storeParticle

// Both store functions create data and use storeChunk to write data
auto A = createData<double>( blockSize, value, 0.0001 ) ;
compA.storeChunk( A, meshOffset, meshExtent );
```
Create data returns a shared pointer `shared_ptr` of given size  with given type & default value.
```c++
template<typename T>
std::shared_ptr< T > createData(const unsigned long& size,  const T& val, const T& increment)

auto E = std::shared_ptr< T > {
    new T[size], []( T * d ) {delete[] d;}
};
    
for(unsigned long  i = 0ul; i < size; i++ )
{
    if (increment != 0)
      E.get()[i] = val+i*increment;
    else
      E.get()[i] = val;
}
```

The `storeChunk` function (*inside the openPMD library*)
```c++
std::shared_ptr< std::queue< IOTask > > m_chunks;

template< typename T >
    void storeChunk(std::shared_ptr< T > data, Offset o, Extent e);

    Parameter< Operation::WRITE_DATASET > dWrite;
    dWrite.offset = o;
    dWrite.extent = e;
    dWrite.dtype = dtype;
    dWrite.data = std::static_pointer_cast< void const >(data);
    m_chunks->push(IOTask(this, dWrite));
```

### Running on Summit

This benchmark writes a few meshes and particles, either 1D, 2D or 3D.

The meshes are viewed as grid of mini blocks. As an example, the mini blocks dimension can be [16, 32, 32].

Next we define the grid based on the mini block. say, [32, 32, 16]. Then our actual mesh size is [16x32, 32x32, 32x16].

Here is a sample input file (“w.input”):
```
dim=3
balanced=true
ratio=1
steps=10
minBlock=16 32 32
grid=32 32 16
```

With the above input file, we will create an openPMD file with the above mesh using
- 3D data
- balanced load
- particle to mesh ratio = 1
- 10 iteration steps

*Note: All files generated are group based. i.e. One file per iteration.*

To run:

```
./8a_benchmark_write_parallel w.input
```

then the file generated are: `../samples/8a_parallel_3Db_*`

### Changes to the write benchmark

Adding a new function `createCUDAData` that will return a shared pointer to a GPU buffer.

```c++
template<typename T>
__global__ void update_array(T *vect, T val, T increment) {
    int i = blockIdx.x;
    vect[i] = val + i * increment;
}

template<typename T>
std::shared_ptr< T > createCUDAData(const unsigned long& size,  const T& val, const T& increment)
  {
    void *E;

    cudaMalloc(&E, size * sizeof(T));
    cudaMemset(E, 0, size * sizeof(T));
    update_array<<<size,1>>>((T *) E, val, increment);

    return std::shared_ptr< T > {
        (T *) E, []( T *d ) {cudaFree(d);}
    };
  }
```

The initial `createData` will still allocate and initialize data on the GPU but will return a CPU pointer with the data.

```c++
template<typename T>
std::shared_ptr< T > createData(const unsigned long& size,  const T& val, const T& increment)
  {
    void *E;

    cudaMalloc(&E, size * sizeof(T));
    cudaMemset(E, 0, size * sizeof(T));
    update_array<<<size,1>>>((T *) E, val, increment);

    T *hostPtr = (T *) malloc(size * sizeof(T));
    cudaMemcpy(hostPtr, E, size * sizeof(T), cudaMemcpyDeviceToHost);

    return std::shared_ptr< T > {
        hostPtr, []( T *d ) {free(d);}
    };
}
```

Compile the new example using CUDA.

```diff
diff --git a/CMakeLists.txt b/CMakeLists.txt
index 9f8e0d4..c34d1df 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -9,6 +9,7 @@ set(openPMD_STANDARD_VERSION 1.1.0)

 list(APPEND CMAKE_MODULE_PATH "${openPMD_SOURCE_DIR}/share/openPMD/cmake")

+find_package(CUDA REQUIRED)

 # CMake policies ##############################################################
 #
@@ -755,6 +756,11 @@ set(openPMD_EXAMPLE_NAMES
     10_streaming_read
     12_span_write
 )
+
+set(openPMD_EXAMPLE_CUDA
+    13_benchmark_write_cuda
+)
+
 set(openPMD_PYTHON_EXAMPLE_NAMES
     2_read_serial
     2a_read_thetaMode_serial
@@ -835,6 +841,16 @@ if(openPMD_BUILD_EXAMPLES)
     endforeach()
 endif()

+if(openPMD_BUILD_EXAMPLES)
+    enable_language(CUDA)
+    foreach(examplename ${openPMD_EXAMPLE_CUDA})
+               add_executable(${examplename} examples/${examplename}.cu)
+               target_link_libraries(${examplename} PRIVATE openPMD ${CUDA_LIBRARIES})
+               target_include_directories(${examplename} PUBLIC ${CUDA_INCLUDE_DIRS})
+        set_target_properties(${examplename} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
+    endforeach()
+endif()
+
```
