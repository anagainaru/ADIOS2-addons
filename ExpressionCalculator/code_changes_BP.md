# Compute data and/or metadata for a derive variable

Building ADIOS2 with derived variable support will allow to create and query derived variables.

```bash
cmake -D ADIOS2_USE_Derived=ON ..
```

## Write side

Derived variables are defined by users giving:
- a string expression
- a mapping between variables in the string expression and ADIOS variables
- the type of derived variable

Library inteface:
```c++
VariableDerived &DefineDerivedVariable(
	const std::string &name,
	const std::string &exp_string,
	const DerivedVarType varType = DerivedVarType::MetadataOnly)

enum class DerivedVarType
{
    MetadataOnly,     ///< Store only the metadata (default)
    ExpressionString, ///< Store only the expression string
    StoreData         ///< Store data and metadata
};
```

Example user code:
```c++
auto deriveVar = bpIO.DefineDerivedVariable(
        "derive/magU",                       // name
        "x : sim/Ux \n”                      // mapping
        "y : sim/Uy \n”
        ”sqrt(x^2 + y^2)",                   // expression
        adios2::DerivedVarType::StoreData    // type of variable
); 
```

The grammar converts the expression into a tree that is being stored by the derived variable. 
The expression tree holds variables in the leafs and operators in the internal nodes. Operators have functions associated with them that can compute the output dimensions and data based on Variable dimensions and data.

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
