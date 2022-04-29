# BP5 Writer 

The information about memory space is saved inside the Variable class. In order to check if a buffer is allocated on the GPU or Host, the `m_MemorySpace` can be used.
```c++
bool isCudaBuffer = (variable.m_MemorySpace == MemorySpace::CUDA);
```

The BP5Writer function is computing the metadata and adding the data from the user buffer to internal ADIOS buffers (wither by memcpy or by saving the pointer).
The Marshal function is called on the input buffer. This function prepares the metadata (including the minmax) and adds the user buffer to the IO vector.
```c++
line 87:            m_BP5Serializer.Marshal((void *)&variable, variable.m_Name.c_str(),
                                    variable.m_Type, variable.m_ElementSize,
                                    DimCount, Shape, Count, Start, values, sync,
                                    nullptr);
```
If the user data is small, the data is directly copied to an internal ADIOS buffer, otherwise only the pointer is saved in the IO vector. This is done by setting the `sync` variable to True or False. For GPU data we do not want to save the pointer, so we set the sync flag to true.

```diff
diff --git a/source/adios2/engine/bp5/BP5Writer.tcc b/source/adios2/engine/bp5/BP5Writer.tcc
index af19b388d..fb4ce58ec 100644
--- a/source/adios2/engine/bp5/BP5Writer.tcc
+++ b/source/adios2/engine/bp5/BP5Writer.tcc
@@ -26,6 +26,10 @@ void BP5Writer::PutCommon(Variable<T> &variable, const T *values, bool sync)
         BeginStep(StepMode::Update);
     }
     variable.SetData(values);
+   // if the user buffer is allocated on the GPU always use sync mode
+   bool isCudaBuffer = (variable.m_MemorySpace == MemorySpace::CUDA);
+   if (isCudaBuffer)
+       sync = true;

     size_t *Shape = NULL;
     size_t *Start = NULL;
```
    
## MinMax

The MinMax function will receive a parameter with the memory space, and will compute the min and max on the GPU is the user buffer is allocated on the device.

```diff
diff --git a/source/adios2/toolkit/format/bp5/BP5Serializer.cpp b/source/adios2/toolkit/format/bp5/BP5Serializer.cpp
index 019f1a8ce..b9dee2756 100644
--- a/source/adios2/toolkit/format/bp5/BP5Serializer.cpp
+++ b/source/adios2/toolkit/format/bp5/BP5Serializer.cpp
@@ -570,15 +570,26 @@ void BP5Serializer::DumpDeferredBlocks(bool forceCopyDeferred)
 }

 static void GetMinMax(const void *Data, size_t ElemCount, const DataType Type,
-                      core::Engine::MinMaxStruct &MinMax)
+                      core::Engine::MinMaxStruct &MinMax, MemorySpace MemSpace)
 {
-
     MinMax.Init(Type);
     if (ElemCount == 0)
         return;
     if (Type == DataType::Compound)
     {
     }
+#ifdef ADIOS2_HAVE_CUDA
+#define pertype(T, N)                                                          \
+   else if (MemSpace == MemorySpace::CUDA && Type == helper::GetDataType<T>())\
+   {                                                                          \
+       const size_t size = ElemCount * sizeof(T);                             \
+       const T *values = (const T *)Data;                                     \
+       helper::CUDAMinMax(values, ElemCount, MinMax.MinUnion.field_##N,       \
+                          MinMax.MaxUnion.field_##N);                         \
+    }
+    ADIOS2_FOREACH_MINMAX_STDTYPE_2ARGS(pertype)
+#undef pertype
+#endif
 #define pertype(T, N)                                                          \
     else if (Type == helper::GetDataType<T>())                                 \
     {                                                                          \
@@ -669,7 +680,8 @@ void BP5Serializer::Marshal(void *Variable, const char *Name,
         MinMax.Init(Type);
         if ((m_StatsLevel > 0) && !Span)
         {
-            GetMinMax(Data, ElemCount, (DataType)Rec->Type, MinMax);
+            GetMinMax(Data, ElemCount, (DataType)Rec->Type, MinMax,
+                     VB->m_MemorySpace);
         }

         if (Rec->OperatorType)
```


## Save data to internal buffers

Update the existing functions to memcpy the buffers from the GPU to work with the format used by BP5 to allocate internal buffers.

```diff
diff --git a/source/adios2/helper/adiosMemory.h b/source/adios2/helper/adiosMemory.h
index a67abc904..e3cc9bf7a 100644
--- a/source/adios2/helper/adiosMemory.h
+++ b/source/adios2/helper/adiosMemory.h
@@ -46,6 +46,9 @@ void InsertToBuffer(std::vector<char> &buffer, const T *source,
 template <class T>
 void CopyFromGPUToBuffer(std::vector<char> &buffer, size_t &position,
                          const T *source, const size_t elements = 1) noexcept;
+template <class T>
+void CudaMemCopyToBuffer(char *buffer, size_t position,
+                         const T *source, const size_t size) noexcept;

 /**
  * Wrapper around cudaMemcpy needed for isolating CUDA interface dependency
  
diff --git a/source/adios2/helper/adiosMemory.inl b/source/adios2/helper/adiosMemory.inl
index cfc15ef97..efdb878be 100644
--- a/source/adios2/helper/adiosMemory.inl
+++ b/source/adios2/helper/adiosMemory.inl
@@ -79,10 +79,17 @@ template <class T>
 void CopyFromGPUToBuffer(std::vector<char> &buffer, size_t &position,
                          const T *source, const size_t elements) noexcept
 {
-    const char *src = reinterpret_cast<const char *>(source);
-    MemcpyGPUToBuffer(buffer.data() + position, src, elements * sizeof(T));
+   CudaMemCopyToBuffer(buffer.data(), position, source, elements * sizeof(T));
     position += elements * sizeof(T);
 }
+
+template <class T>
+void CudaMemCopyToBuffer(char *buffer, size_t position,
+                         const T *source, const size_t size) noexcept
+{
+    const char *src = reinterpret_cast<const char *>(source);
+    MemcpyGPUToBuffer(buffer + position, src, size);
+}
 #endif

 template <class T>
```

Update the Marchal function to send information about the memory space when using AddToVec to save some user data.

```diff
diff --git a/source/adios2/toolkit/format/bp5/BP5Serializer.cpp b/source/adios2/toolkit/format/bp5/BP5Serializer.cpp
@@ -700,7 +712,7 @@ void BP5Serializer::Marshal(void *Variable, const char *Name,
             {
                 DataOffset = m_PriorDataBufferSizeTotal +
                              CurDataBuffer->AddToVec(ElemCount * ElemSize, Data,
-                                                     ElemSize, Sync);
+                                                     ElemSize, Sync, VB->m_MemorySpace);
             }
         }
         else
```

The `AddToVec` function is a virtual function defined in `ChunkV.cpp`, in `BufferV.cpp` and `MallocV.cpp` files. All functions signatures need to change to include the memory space.
```diff
diff --git a/source/adios2/toolkit/format/buffer/BufferV.h b/source/adios2/toolkit/format/buffer/BufferV.h
index b55a9dabb..db6bab470 100644
--- a/source/adios2/toolkit/format/buffer/BufferV.h
+++ b/source/adios2/toolkit/format/buffer/BufferV.h
@@ -41,7 +41,7 @@ public:
     virtual void Reset();

     virtual size_t AddToVec(const size_t size, const void *buf, size_t align,
-                            bool CopyReqd) = 0;
+                            bool CopyReqd, MemorySpace MemSpace=MemorySpace::Host) = 0;

     struct BufferPos
     {

diff --git a/source/adios2/toolkit/format/buffer/chunk/ChunkV.h b/source/adios2/toolkit/format/buffer/chunk/ChunkV.h
index 4db67aed8..14ededfb1 100644
--- a/source/adios2/toolkit/format/buffer/chunk/ChunkV.h
+++ b/source/adios2/toolkit/format/buffer/chunk/ChunkV.h
@@ -32,7 +32,7 @@ public:
     virtual std::vector<core::iovec> DataVec() noexcept;

     virtual size_t AddToVec(const size_t size, const void *buf, size_t align,
-                            bool CopyReqd);
+                            bool CopyReqd, MemorySpace MemSpace=MemorySpace::Host);

     virtual BufferPos Allocate(const size_t size, size_t align);
     virtual void DownsizeLastAlloc(const size_t oldSize, const size_t newSize);
@@ -40,6 +40,8 @@ public:
     virtual void *GetPtr(int bufferIdx, size_t posInBuffer);

     void CopyExternalToInternal();
+   void CopyDataToBuffer(const size_t size, const void *buf, size_t pos,
+                          MemorySpace MemSpace);

 private:
     std::vector<char *> m_Chunks;
     
diff --git a/source/adios2/toolkit/format/buffer/malloc/MallocV.cpp b/source/adios2/toolkit/format/buffer/malloc/MallocV.cpp
index 7f0015a62..c23cb1e41 100644
--- a/source/adios2/toolkit/format/buffer/malloc/MallocV.cpp
+++ b/source/adios2/toolkit/format/buffer/malloc/MallocV.cpp
@@ -85,7 +85,7 @@ void MallocV::CopyExternalToInternal()
 }

 size_t MallocV::AddToVec(const size_t size, const void *buf, size_t align,
-                         bool CopyReqd)
+                         bool CopyReqd, MemorySpace MemSpace)
 {
     if (size == 0)
     {
diff --git a/source/adios2/toolkit/format/buffer/malloc/MallocV.h b/source/adios2/toolkit/format/buffer/malloc/MallocV.h
index ce1d43ccf..4cf8d2702 100644
--- a/source/adios2/toolkit/format/buffer/malloc/MallocV.h
+++ b/source/adios2/toolkit/format/buffer/malloc/MallocV.h
@@ -36,7 +36,7 @@ public:
     virtual void Reset();

     virtual size_t AddToVec(const size_t size, const void *buf, size_t align,
-                            bool CopyReqd);
+                            bool CopyReqd, MemorySpace MemSpace=MemorySpace::Host);

     virtual BufferPos Allocate(const size_t size, size_t align);
     void DownsizeLastAlloc(const size_t oldSize, const size_t newSize);
```

The function called from Marchal is defined in `ChunkV.cpp`. If `CopyReqd` is set to False (large vectors, deferred mode), only a copy to the buffer is kept.
```c++
    if (!CopyReqd && !m_AlwaysCopy)
    {
        // just add buf to internal version of output vector
        VecEntry entry = {true, buf, 0, size};
        DataV.push_back(entry);
    }
```
Otherwise copy it to internal buffers. For GPU buffers sync is set to true so all user buffers are copied to internal buffers regardless of their size.
```diff
diff --git a/source/adios2/toolkit/format/buffer/chunk/ChunkV.cpp b/source/adios2/toolkit/format/buffer/chunk/ChunkV.cpp
index bf1e3a013..d39817ac3 100644
--- a/source/adios2/toolkit/format/buffer/chunk/ChunkV.cpp
+++ b/source/adios2/toolkit/format/buffer/chunk/ChunkV.cpp
@@ -8,6 +8,7 @@

 #include "ChunkV.h"
 #include "adios2/toolkit/format/buffer/BufferV.h"
+#include "adios2/helper/adiosFunctions.h"

 #include <algorithm>
 #include <assert.h>
@@ -82,7 +83,7 @@ void ChunkV::CopyExternalToInternal()
 }

 size_t ChunkV::AddToVec(const size_t size, const void *buf, size_t align,
-                        bool CopyReqd)
+                        bool CopyReqd, MemorySpace MemSpace)
 {
     if (size == 0)
     {
@@ -120,7 +121,8 @@ size_t ChunkV::AddToVec(const size_t size, const void *buf, size_t align,
         if (AppendPossible)
         {
             // We can use current chunk, just append the data;
-            memcpy(m_TailChunk + m_TailChunkPos, buf, size);
+           CopyDataToBuffer(size, buf, m_TailChunkPos, MemSpace);
             DataV.back().Size += size;
             m_TailChunkPos += size;
         }
@@ -132,7 +134,7 @@ size_t ChunkV::AddToVec(const size_t size, const void *buf, size_t align,
                 NewSize = size;
             m_TailChunk = (char *)malloc(NewSize);
             m_Chunks.push_back(m_TailChunk);
-            memcpy(m_TailChunk, buf, size);
+           CopyDataToBuffer(size, buf, 0, MemSpace);
             m_TailChunkPos = size;
             VecEntry entry = {false, m_TailChunk, 0, size};
             DataV.push_back(entry);
@@ -142,6 +144,18 @@ size_t ChunkV::AddToVec(const size_t size, const void *buf, size_t align,
     return retOffset;
 }

+void ChunkV::CopyDataToBuffer(const size_t size, const void *buf, size_t pos,
+                             MemorySpace MemSpace){
+#ifdef ADIOS2_HAVE_CUDA
+    if(MemSpace == MemorySpace::CUDA)
+   {
+       helper::CudaMemCopyToBuffer(m_TailChunk, pos, buf, size);
+       return;
+   }
+#endif
+   memcpy(m_TailChunk + pos, buf, size);
+}
+
 BufferV::BufferPos ChunkV::Allocate(const size_t size, size_t align)
 {
     if (size == 0)
```


## Update the CUDA example

To use the BP5 engine. BP4 was reading one step of metadata on Open which allowed the code to call InquireVariable outside the BeginStep - EndStep block. BP5 doesn't do this anymore, so all metadata enquires need to take place inside the for loop.

```diff
diff --git a/examples/cuda/cudaWriteRead.cu b/examples/cuda/cudaWriteRead.cu
index e83754ac7..433dd5814 100644
--- a/examples/cuda/cudaWriteRead.cu
+++ b/examples/cuda/cudaWriteRead.cu
@@ -26,6 +26,7 @@ int BPWrite(const std::string fname, const size_t N, int nSteps){
   // Set up the ADIOS structures
   adios2::ADIOS adios;
   adios2::IO io = adios.DeclareIO("WriteIO");
+  io.SetEngine("BP5");

   // Declare an array for the ADIOS data of size (NumOfProcesses * N)
   const adios2::Dims shape{static_cast<size_t>(N)};
@@ -61,28 +62,23 @@ int BPRead(const std::string fname, const size_t N, int nSteps){
   // Create ADIOS structures
   adios2::ADIOS adios;
   adios2::IO io = adios.DeclareIO("ReadIO");
+  io.SetEngine("BP5");

   adios2::Engine bpReader = io.Open(fname, adios2::Mode::Read);

-  auto data = io.InquireVariable<float>("data");
-  std::cout << "Steps expected by the reader: " << bpReader.Steps() << std::endl;
-  std::cout << "Expecting data per step: " << data.Shape()[0];
-  std::cout  << " elements" << std::endl;
-
-  int write_step = bpReader.Steps();
-  // Create the local buffer and initialize the access point in the ADIOS file
-  std::vector<float> simData(N); //set size to N
-  const adios2::Dims start{0};
-  const adios2::Dims count{N};
-  const adios2::Box<adios2::Dims> sel(start, count);
-  data.SetSelection(sel);
-
-  // Read the data in each of the ADIOS steps
-  for (size_t step = 0; step < write_step; step++)
+  unsigned int step = 0;
+  for (; bpReader.BeginStep() == adios2::StepStatus::OK; ++step)
   {
-      data.SetStepSelection({step, 1});
+     auto data = io.InquireVariable<float>("data");
+     // Create the local buffer and initialize the access point in the ADIOS file
+     std::vector<float> simData(N); //set size to N
+     const adios2::Dims start{0};
+     const adios2::Dims count{N};
+     const adios2::Box<adios2::Dims> sel(start, count);
+     data.SetSelection(sel);
+
       bpReader.Get(data, simData.data());
-      bpReader.PerformGets();
+      bpReader.EndStep();
       std::cout << "Simualation step " << step << " : ";
       std::cout << simData.size() << " elements: " << simData[1] << std::endl;
   }
```
