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
