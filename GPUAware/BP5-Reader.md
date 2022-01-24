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
The actual copy of data into the user buffer is being done in the `PerformGets` function while the `QueueGet` fills a structure with all the pending read requests.

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

Changes to this code are needed to allow the read function to use a memory space.
```diff
diff --git a/source/adios2/toolkit/format/bp5/BP5Deserializer.cpp b/source/adios2/toolkit/format/bp5/BP5Deserializer.cpp
index 6f89265ec..ba851e308 100644
--- a/source/adios2/toolkit/format/bp5/BP5Deserializer.cpp
+++ b/source/adios2/toolkit/format/bp5/BP5Deserializer.cpp
 @@ -927,6 +928,7 @@ bool BP5Deserializer::QueueGetSingle(core::VariableBase &variable,
         Req.Count = variable.m_Count;
         Req.Start = variable.m_Start;
         Req.Step = Step;
+        Req.MemSpace = variable.m_MemorySpace;
         Req.Data = DestData;
         PendingRequests.push_back(Req);
     }
diff --git a/source/adios2/toolkit/format/bp5/BP5Deserializer.h b/source/adios2/toolkit/format/bp5/BP5Deserializer.h
index 86be0d40a..fc22cf7a9 100644
--- a/source/adios2/toolkit/format/bp5/BP5Deserializer.h
+++ b/source/adios2/toolkit/format/bp5/BP5Deserializer.h
@@ -200,6 +200,7 @@ private:
         size_t BlockID;
         Dims Start;
         Dims Count;
+        MemorySpace MemSpace;
         void *Data;
     };
     std::vector<BP5ArrayRequest> PendingRequests;
```

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

The `GenerateReadRequests` function allocates memory for internal buffers in the `ReadRequests` structure and `ReadData` reads data from file to the internal buffers. The `FinalizeGets` function uses two functions `ExtractSelectionFromPartial` (row or column major RM/CM) to copy the data to the user buffers.

Two changes are needed, first to pass the memory space to these two functions.

```diff
diff --git a/source/adios2/toolkit/format/bp5/BP5Deserializer.cpp b/source/adios2/toolkit/format/bp5/BP5Deserializer.cpp
index 6f89265ec..ba851e308 100644
--- a/source/adios2/toolkit/format/bp5/BP5Deserializer.cpp
+++ b/source/adios2/toolkit/format/bp5/BP5Deserializer.cpp
@@ -10,6 +10,7 @@
 #include "adios2/core/Engine.h"
 #include "adios2/core/IO.h"
 #include "adios2/core/VariableBase.h"
+#include "adios2/helper/adiosFunctions.h"

 #include "BP5Deserializer.h"
 #include "BP5Deserializer.tcc"
@@ -1168,14 +1170,14 @@ void BP5Deserializer::FinalizeGets(std::vector<ReadRequest> Requests)
                         ExtractSelectionFromPartialRM(
                             ElementSize, DimCount, GlobalDimensions, RankOffset,
                             RankSize, SelOffset, SelSize, IncomingData,
-                            (char *)Req.Data);
+                            (char *)Req.Data, Req.MemSpace);
                     }
                     else
                     {
                         ExtractSelectionFromPartialCM(
                             ElementSize, DimCount, GlobalDimensions, RankOffset,
                             RankSize, SelOffset, SelSize, IncomingData,
-                            (char *)Req.Data);
+                            (char *)Req.Data, Req.MemSpace);
                     }
                 }
             }
@@ -1257,9 +1272,9 @@
 // Row major version
 void BP5Deserializer::ExtractSelectionFromPartialRM(
     int ElementSize, size_t Dims, const size_t *GlobalDims,
     const size_t *PartialOffsets, const size_t *PartialCounts,
     const size_t *SelectionOffsets, const size_t *SelectionCounts,
-    const char *InData, char *OutData)
+    const char *InData, char *OutData, MemorySpace MemSpace)
 {
     size_t BlockSize;
     size_t SourceBlockStride = 0;
@@ -1353,7 +1368,7 @@ void BP5Deserializer::ExtractSelectionFromPartialCM(
     int ElementSize, size_t Dims, const size_t *GlobalDims,
     const size_t *PartialOffsets, const size_t *PartialCounts,
     const size_t *SelectionOffsets, const size_t *SelectionCounts,
-    const char *InData, char *OutData)
+    const char *InData, char *OutData, MemorySpace MemSpace)
 {
     int BlockSize;
     int SourceBlockStride = 0;
diff --git a/source/adios2/toolkit/format/bp5/BP5Deserializer.h b/source/adios2/toolkit/format/bp5/BP5Deserializer.h
index 86be0d40a..fc22cf7a9 100644
--- a/source/adios2/toolkit/format/bp5/BP5Deserializer.h
+++ b/source/adios2/toolkit/format/bp5/BP5Deserializer.h
@@ -171,20 +171,20 @@ private:
     bool GetSingleValueFromMetadata(core::VariableBase &variable,
                                     BP5VarRec *VarRec, void *DestData,
                                     size_t Step, size_t WriterRank);
    void ExtractSelectionFromPartialRM(
        int ElementSize, size_t Dims, const size_t *GlobalDims,
        const size_t *PartialOffsets, const size_t *PartialCounts,
        const size_t *SelectionOffsets, const size_t *SelectionCounts,
        const char *InData, char *OutData,
+        MemorySpace MemSpace = MemorySpace::Host);
    void ExtractSelectionFromPartialCM(
        int ElementSize, size_t Dims, const size_t *GlobalDims,
        const size_t *PartialOffsets, const size_t *PartialCounts,
        const size_t *SelectionOffsets, const size_t *SelectionCounts,
        const char *InData, char *OutData,
+        MemorySpace MemSpace = MemorySpace::Host);

     enum RequestTypeEnum
     {
```

**Copy data from internal buffers to the user buffer**

Second change needed is to separate the logic for copying the data so that GPU buffers use cudamemcpy instead of memcpy.

```diff
diff --git a/source/adios2/toolkit/format/bp5/BP5Deserializer.cpp b/source/adios2/toolkit/format/bp5/BP5Deserializer.cpp
index befbf1ba7..ba851e308 100644
--- a/source/adios2/toolkit/format/bp5/BP5Deserializer.cpp
+++ b/source/adios2/toolkit/format/bp5/BP5Deserializer.cpp
@@ -1256,6 +1256,19 @@ static int FindOffsetCM(size_t Dims, const size_t *Size, const size_t *Index)
  * *******************************
  */

+void BP5Deserializer::MemCopyData(char *OutData, const char *InData,
+                                  size_t Size, MemorySpace MemSpace)
+{
+#ifdef ADIOS2_HAVE_CUDA
+    if (MemSpace == MemorySpace::CUDA)
+    {
+        helper::CudaMemCopyToBuffer(OutData, 0, InData, Size);
+        return;
+    }
+#endif
+    memcpy(OutData, InData, Size);
+}
+
 // Row major version
 void BP5Deserializer::ExtractSelectionFromPartialRM(
     int ElementSize, size_t Dims, const size_t *GlobalDims,
@@ -1343,7 +1356,7 @@ void BP5Deserializer::ExtractSelectionFromPartialRM(
     size_t i;
     for (i = 0; i < BlockCount; i++)
     {
-        memcpy(OutData, InData, BlockSize * ElementSize);
+        MemCopyData(OutData, InData, BlockSize * ElementSize, MemSpace);
         InData += SourceBlockStride;
         OutData += DestBlockStride;
     }
@@ -1444,7 +1457,7 @@ void BP5Deserializer::ExtractSelectionFromPartialCM(
     OutData += DestBlockStartOffset;
     for (int i = 0; i < BlockCount; i++)
     {
-        memcpy(OutData, InData, BlockSize * ElementSize);
+        MemCopyData(OutData, InData, BlockSize * ElementSize, MemSpace);
         InData += SourceBlockStride;
         OutData += DestBlockStride;
     }
diff --git a/source/adios2/toolkit/format/bp5/BP5Deserializer.h b/source/adios2/toolkit/format/bp5/BP5Deserializer.h
index 15f3c2e10..fc22cf7a9 100644
--- a/source/adios2/toolkit/format/bp5/BP5Deserializer.h
+++ b/source/adios2/toolkit/format/bp5/BP5Deserializer.h
@@ -171,6 +171,8 @@ private:
     bool GetSingleValueFromMetadata(core::VariableBase &variable,
                                     BP5VarRec *VarRec, void *DestData,
                                     size_t Step, size_t WriterRank);
+    void MemCopyData(char *OutData, const char *InData, size_t Size,
+                     MemorySpace MemSpace);
     void ExtractSelectionFromPartialRM(
         int ElementSize, size_t Dims, const size_t *GlobalDims,
         const size_t *PartialOffsets, const size_t *PartialCounts,
```

## Changes to the CUDA example

The code is updated to use GPU buffers for both the read and the 
```diff
diff --git a/examples/cuda/cudaWriteRead.cu b/examples/cuda/cudaWriteRead.cu
index 433dd5814..0caccd212 100644
--- a/examples/cuda/cudaWriteRead.cu
+++ b/examples/cuda/cudaWriteRead.cu
@@ -171,6 +171,8 @@ private:
    adios2::Engine bpReader = io.Open(fname, adios2::Mode::Read);

    unsigned int step = 0;
+    float *gpuSimData;
+    cudaMalloc(&gpuSimData, N * sizeof(float));
+    cudaMemset(gpuSimData, 0, N);
    for (; bpReader.BeginStep() == adios2::StepStatus::OK; ++step)
    {
        auto data = io.InquireVariable<float>("data");
@@ -191,6 +191,8 @@ private:
        const adios2::Box<adios2::Dims> sel(start, count);
        data.SetSelection(sel);

+        data.SetMemorySpace(adios2::MemorySpace::CUDA);
+        bpReader.Get(data, gpuSimData, adios2::Mode::Deferred);
        bpReader.EndStep();
+        cudaMemcpy(simData.data(), gpuSimData, N, cudaMemcpyDeviceToHost);
        std::cout << "Simualation step " << step << " : ";
        std::cout << simData.size() << " elements: " << simData[1] << std::endl;
```
