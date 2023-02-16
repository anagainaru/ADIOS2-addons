## Metadata

Computating the min and max has a separate logic for BP4 and BP5 for Host buffers.

1. In `source/adios2/toolkit/format/bp5/BP5Serializer.cpp` used by BP5
```c++
static void GetMinMax(const void *Data, size_t ElemCount, const DataType Type,
                      MinMaxStruct &MinMax, MemorySpace MemSpace)
{
  ...
  auto res = std::minmax_element(values, values + ElemCount);
  ...
}
```

2. In `source/adios2/helper/adiosMath.inl` used by BP4 and BP3
```c++
template <class T>
inline void GetMinMax(const T *values, const size_t size, T &min,
                      T &max, MemorySpace MemSpace) noexcept
{
    auto bounds = std::minmax_element(values, values + size);
    min = *bounds.first;
    max = *bounds.second;
}
```

For GPU pointers the `GPUMinMax` function is used (defined in `source/adios2/helper/adiosMath.inl`) and called from the GetMinMax functions.
```c++
template <class T>
void GPUMinMax(const T *values, const size_t size, T &min, T &max)
{
#ifdef ADIOS2_HAVE_CUDA
    CUDAMinMax(values, size, min, max);
#endif
}
```

## Copy to ADIOS buffers
