## Compute data and/or metadata for a derive variable

**BP4** 
The user data is stored in blocks inside the `Variable` class.
```c++
    struct BPInfo
    {
        T *Data = nullptr;
    }
```
Each variable can receive multiple blocks in the same step.

**BP5**
The user data is stored in blocks inside the `BP5Serializer` class in the `DeferredExterns` structure. 
Each variable can receive multiple blocks in the same step that will be merged into the structure. 
There is no information in this structure about the variable it came from (it needs to be retrieved from the metadata).

```c++
    struct DeferredExtern
    {
        size_t MetaOffset;
        size_t BlockID;
        const void *Data;
        size_t DataSize;
        size_t AlignReq;
    };
    std::vector<DeferredExtern> DeferredExterns;
```
