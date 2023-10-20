## Compute data and/or metadata for a derive variable

Building ADIOS2 with derived variable support will allow to create and query derived variables.

```bash
cmake -D ADIOS2_USE_Derived=ON ..
```

**Write side**

Creating derived variables is given in the following diagram:
<img width="304" alt="Screenshot 2023-08-28 at 3 25 49 PM" src="https://github.com/anagainaru/ADIOS2-addons/assets/16229479/9b73177a-40c5-41bd-88ed-f10fea2b4c83">

Derived variables are defined by users given a string expression. The grammer converts the expression into a tree that is being stored by the derived variable. The expression class keeps track of all the pointers to blocks of data that are being `Put` in each step.

For defered mode the data pointers are directly the user buffer. For sync, the data is taken from the BP4/BP5 buffers:

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

Writing derived variables is given in the following diagram:
<img width="365" alt="Screenshot 2023-08-28 at 3 44 24 PM" src="https://github.com/anagainaru/ADIOS2-addons/assets/16229479/1b7bb96b-4b84-428a-a273-ec8f884010f7">

On EndStep, the data for all derived variables is computed using the buffers stored in the expression attached to all derived variables. The new data is sent to the engine that is responsible with writing the data/metadata or only the metadata based on the derived variable type.
