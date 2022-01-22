# BP5 Reader

There are two ways of reading data from a BP file to a buffer provided by the user: Sync and Deferred.
Ths functions that are called for this are implemented in `adios2 > engine > bp5 > BP5Reader.cpp` and `BP5Reader.tcc`.

```c++
template <class T>
inline void BP5Reader::GetSyncCommon(Variable<T> &variable, T *data)
{
    bool need_sync = m_BP5Deserializer->QueueGet(variable, data);
    if (need_sync)
        PerformGets();
}

GetDeferredCommon
template <class T>
void BP5Reader::GetDeferredCommon(Variable<T> &variable, T *data)
{
    (void)m_BP5Deserializer->QueueGet(variable, data);
}
```

Both versions call `QueueGet` on the variable and user buffer. 
The sync version calls `PerformGets` while the deferred mode postpones calling the function until the user calls it or until EndStep.

## QueueGet

The QueueGet function is implemented in the BP5Serializer

```c++
bool BP5Deserializer::QueueGet(core::VariableBase &variable, void *DestData)

bool BP5Deserializer::QueueGetSingle(core::VariableBase &variable,
                                     void *DestData, size_t Step)
{
        // different logic for single values

        BP5ArrayRequest Req;
        Req.VarRec = VarRec;
        Req.RequestType = Global;
        Req.BlockID = variable.m_BlockID;
        Req.Count = variable.m_Count;
        Req.Start = variable.m_Start;
        Req.Step = Step;
        Req.Data = DestData;
        PendingRequests.push_back(Req);
}
```

Requests are pushed into a queue of requests and the data pointer to the user buffer is saved in the `Data` field.

## PerformGets 

Declared in `BP5Reader.cpp`, the function is filling the buffers with data. 

```c++
void BP5Reader::PerformGets()
{
    auto ReadRequests = m_BP5Deserializer->GenerateReadRequests();
    // Potentially optimize read requests, make contiguous, etc.
    for (const auto &Req : ReadRequests)
    {
        ReadData(Req.WriterRank, Req.Timestep, Req.StartOffset, Req.ReadLength,
                 Req.DestinationAddr);
    }

    m_BP5Deserializer->FinalizeGets(ReadRequests);
}
```

The `GenerateReadRequests` function allocates memory for internal buffers in the `ReadRequests` structure and `ReadData` reads data from file to the internal buffers.


```c++
ReadData
    ThisDataPos = helper::ReadValue<uint64_t>(
        m_MetadataIndex.m_Buffer, ThisFlushInfo, m_Minifooter.IsLittleEndian);
    m_DataFileManager.ReadFile(Destination, RemainingLength, ThisDataPos,
                               SubfileNum);
```

Adios2 > helper > adiosMemory.inl (thrust::copy instead of std::copy)
```c++
void BP5Deserializer::FinalizeGets(std::vector<ReadRequest> Requests)
{
    for (const auto &Req : PendingRequests)
    {
    }
}
```
