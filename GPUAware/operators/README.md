# Operators in ADIOS2

<img width="723" alt="Screen Shot 2023-02-24 at 11 18 25 AM" src="https://user-images.githubusercontent.com/16229479/221230759-8f78f5e2-5016-433d-8c9b-dc10d603e04c.png">

Operators need to be GPU-aware inside ADIOS2. There are three cases when an operator uses memory copy within its internal ADIOS2 logic:
1. The operator is not always applied to the buffer (e.g. due to an array size threshold)
2. The buffer for the compressed/decompressed version is allocated by the operator and returned to ADIOS2
3. The operator is composed of a series of data copies 

In all cases the operator can only be applied if the operator supports outputing a CPU buffers (even if the input is GPU). The **IsGPUAware** function will return `true` if the operator can receive a GPU buffer and output a CPU buffer. 

### Threshold for applying compression: Blosc, MGARD, MGARD Plus

```c++
    if (useMemcpy)
    {
        std::memcpy(bufferOut + bufferOutOffset, dataIn + inputOffset, sizeIn);
        bufferOutOffset += sizeIn;
        headerPtr->SetNumChunks(0u);
    }
```

Same bahavior in the logic for Blosc backcompatibility with BP: `source/adios2/toolkit/format/bp/bpBackCompatOperation/compress/BPBackCompatBlosc.cpp`


### Ouput in a separate buffer: SZ, PNG, LibPressio

The compression function takes the input buffer, allocates an internal buffer and returns this in a `void *`.
This buffer is then memcpyed into the ADIOS2 internal buffer before it's freed.

The decompress has a similar behavior, returning the decompressed buffer into a `void *` that needs to be copied into the user buffer.

```c++
    void *result = SZ_decompress(dtype,
                      reinterpret_cast<unsigned char *>(
                          const_cast<char *>(bufferIn + bufferInOffset)),
                      sizeIn - bufferInOffset, 0, convertedDims[0],
                      convertedDims[1], convertedDims[2], convertedDims[3]);
    std::memcpy(dataOut, result, dataSizeBytes);
    free(result);
```

Currently all these compression operators expect CPU allocated buffers for both input and output.

### Others: Sirius, NULL

Contains logic that requires to copy the input data into one or multiple tiers and from the tiers to the output buffer.
Expects CPU allocated buffers.

## Code changes

All operators have a function that returns the header size used by the operator (after applying the compress or decompress logic).

```diff
diff --git a/source/adios2/core/Operator.h b/source/adios2/core/Operator.h
index 0be396dc5..63a33e749 100644
--- a/source/adios2/core/Operator.h
+++ b/source/adios2/core/Operator.h
@@ -68,6 +68,8 @@ public:
                               const std::string &, const size_t, const Dims &,
                               const Dims &, const Dims &) const;

+    virtual size_t GetHeaderSize() const;
+
```

The memcpy logic is moved from the operator code to internal ADIOS2 functions. On the compression side, the serialized calls the operator that applies what was given by the user. If the operator was not applied (return 0) we copy the memory in the internal ADIOS2 buffer.

```diff
diff --git a/source/adios2/toolkit/format/bp/BPSerializer.tcc b/source/adios2/toolkit/format/bp/BPSerializer.tcc
index 3471e8179..ff4f6f9eb 100644
--- a/source/adios2/toolkit/format/bp/BPSerializer.tcc
+++ b/source/adios2/toolkit/format/bp/BPSerializer.tcc
@@ -405,11 +405,17 @@ void BPSerializer::PutOperationPayloadInBuffer(
     const core::Variable<T> &variable,
     const typename core::Variable<T>::BPInfo &blockInfo)
 {
-    const size_t outputSize = blockInfo.Operations[0]->Operate(
+    size_t outputSize = blockInfo.Operations[0]->Operate(
         reinterpret_cast<char *>(blockInfo.Data), blockInfo.Start,
         blockInfo.Count, variable.m_Type,
         m_Data.m_Buffer.data() + m_Data.m_Position);

+    if (outputSize == 0) // the operator was not applied
+        outputSize = helper::CopyMemoryWithOpHeader(
+            reinterpret_cast<char *>(blockInfo.Data), blockInfo.Count,
+            variable.m_Type, m_Data.m_Buffer.data() + m_Data.m_Position,
+            blockInfo.Operations[0]->GetHeaderSize(), blockInfo.MemSpace);
+
     m_Data.m_Position += outputSize;
     m_Data.m_AbsolutePosition += outputSize;
diff --git a/source/adios2/helper/adiosMemory.cpp b/source/adios2/helper/adiosMemory.cpp
index ad7928eae..779cd68ce 100644
--- a/source/adios2/helper/adiosMemory.cpp
+++ b/source/adios2/helper/adiosMemory.cpp
@@ -20,6 +20,16 @@ namespace adios2
 namespace helper
 {

+size_t CopyMemoryWithOpHeader(const char *src, const Dims &blockCount,
+                              const DataType type, char *dest,
+                              size_t destOffset, const MemorySpace memSpace)
+{
+    const size_t sizeIn = GetTotalSize(blockCount, GetDataTypeSize(type));
+    CopyContiguousMemory(src, sizeIn, dest + destOffset,
+                         /* endianReverse */ false, memSpace);
+    return destOffset + sizeIn;
+}
+
```

On the decompress side, the data is read on the internal ADIOS2 buffer then the `OperatorFactory` detects from the header what operator was applied and calls the inverse operator. If the inverse is not applied (return 0) the memory is copied directly in the user buffer.

```diff
diff --git a/source/adios2/operator/OperatorFactory.cpp b/source/adios2/operator/OperatorFactory.cpp
index 1979aeeb0..f6a6ab703 100644
--- a/source/adios2/operator/OperatorFactory.cpp
+++ b/source/adios2/operator/OperatorFactory.cpp
@@ -169,7 +169,7 @@ std::shared_ptr<Operator> MakeOperator(const std::string &type,
 }

 size_t Decompress(const char *bufferIn, const size_t sizeIn, char *dataOut,
-                  std::shared_ptr<Operator> op)
+                  MemorySpace memSpace, std::shared_ptr<Operator> op)
 {
     Operator::OperatorType compressorType;
     std::memcpy(&compressorType, bufferIn, 1);
@@ -177,7 +177,15 @@ size_t Decompress(const char *bufferIn, const size_t sizeIn, char *dataOut,
     {
         op = MakeOperator(OperatorTypeToString(compressorType), {});
     }
-    return op->InverseOperate(bufferIn, sizeIn, dataOut);
+    size_t sizeOut = op->InverseOperate(bufferIn, sizeIn, dataOut);
+    if (sizeOut == 0) // the inverse operator was not applied
+    {
+        size_t headerSize = op->GetHeaderSize();
+        sizeOut = sizeIn - headerSize;
+        helper::CopyContiguousMemory(bufferIn + headerSize, sizeOut, dataOut,
+                                     /*endianReverse*/ false, memSpace);
+    }
+    return sizeOut;
 }
```
