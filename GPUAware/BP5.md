# BP5 Writer 

## Put function

Potential place to save the GPU information `variable.SetData` in `BP5Writer.tcc`
```c++
template <class T>
void BP5Writer::PutCommon(Variable<T> &variable, const T *values, bool sync)
{
    if (!m_BetweenStepPairs)
    {
        BeginStep(StepMode::Update);
    }
    variable.SetData(values);
}
```

In the same function the Marshal function is called on the input buffer. This function prepares the metadata (including the minmax) and adds the user buffer to the IO vector.
```c++
line 87:            m_BP5Serializer.Marshal((void *)&variable, variable.m_Name.c_str(),
                                    variable.m_Type, variable.m_ElementSize,
                                    DimCount, Shape, Count, Start, values, sync,
                                    nullptr);
```
If the user data is small, the data is directly copied to an internal ADIOS buffer, otherwise only the pointer is saved in the IO vector. This is done by setting the `sync` variable to True or False.

In the `PutCommon` function:
```c++
    if (!sync)
    {
        /* If arrays is small, force copying to internal buffer to aggregate
         * small writes */
        size_t n = helper::GetTotalSize(variable.m_Count) * sizeof(T);
        if (n < m_Parameters.MinDeferredSize)
        {
            sync = true;
        }
    }
```

## Marchal serializer

```c++
line 593: void BP5Serializer::Marshal(void *Variable, const char *Name,
                            const DataType Type, size_t ElemSize,
                            size_t DimCount, const size_t *Shape,
                            const size_t *Count, const size_t *Offsets,
                            const void *Data, bool Sync,
                            BufferV::BufferPos *Span)
```
Function computes the metadata for every Put and saves a pointer or copies the data from the user buffer to ADIOS buffers.

**Changes to the code:**
- Step 1. Compute the minmax metadata

```c++
        core::Engine::MinMaxStruct MinMax;
        MinMax.Init(Type);
        if ((m_StatsLevel > 0) && !Span)
        {
            GetMinMax(Data, ElemCount, (DataType)Rec->Type, MinMax);
        }
```

Metadata min max defined in
```c++
line 572: static void GetMinMax(const void *Data, size_t ElemCount, const DataType Type,
                      core::Engine::MinMaxStruct &MinMax)
```
Line 582 computes the minmax for a std::vector

This needs to be replaced by a cuda function in case the value is on the GPU.


- Step 2. Save the user buffer data

```c++
line 701:   DataOffset = m_PriorDataBufferSizeTotal +
                             CurDataBuffer->AddToVec(ElemCount * ElemSize, Data,
                                                     ElemSize, Sync);
```

The `AddToVec` function is defined in `ChunkV.cpp` file (line 84). If `CopyReqd` is set to False (large vectors, deferred mode), only a copy to the buffer is kept.
```c++
    if (!CopyReqd && !m_AlwaysCopy)
    {
        // just add buf to internal version of output vector
        VecEntry entry = {true, buf, 0, size};
        DataV.push_back(entry);
    }
```
Otherwise copy it to internal buffers.
```c++
     if (AppendPossible)
        {
            // We can use current chunk, just append the data;
            memcpy(m_TailChunk + m_TailChunkPos, buf, size);
            DataV.back().Size += size;
            m_TailChunkPos += size;
        }
        else
        {
            // We need a new chunk, get the larger of size or m_ChunkSize
            size_t NewSize = m_ChunkSize;
            if (size > m_ChunkSize)
                NewSize = size;
            m_TailChunk = (char *)malloc(NewSize);
            m_Chunks.push_back(m_TailChunk);
            memcpy(m_TailChunk, buf, size);
            m_TailChunkPos = size;
            VecEntry entry = {false, m_TailChunk, 0, size};
            DataV.push_back(entry);
        }
```
The `memcpy` call needs to be replaced by `cudamemcpy`.


