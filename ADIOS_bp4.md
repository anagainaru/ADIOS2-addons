# ADIOS Workflow for the BP4 Engine

Steps from initialization to performing I/O operations, description of the buffers needed and the format of the headers.
The BP4 engine is a file engine and uses POSIX functions underneath.

<a href="#Buffer Headers" /> Buffer Headers </a> <br/>
<a href="#Debugging ADIOS" /> Debugging ADIOS </a>

## Workflow

### Write

```c++
adios2::Engine bpWriter = io.Open(fname, adios2::Mode::Write);
```

The `Open` function, initializes a Serializer object and a BP4Writer object and initializes all the buffers and objects needed for writing data into a file.
  - BP4Writer::InitParameters 
      - Initalizes input parameters passed through the `adios2::IO` variable (e.g. `io.SetEngine('BP4')`)

  - BP4Writer::InitTransports 
      - Transport is set by default to `File`. The function creates the files it will use to write the data (If burst buffers are used the file names contain the path to the BB)
      - In case there is aggregation (not all ranks write files) the `m_BP4Serializer.m_Aggregator.m_IsConsumer` decides which ranks are in charge of writing
      - The BP folder is created and inside the data.rank files are created
```
m_FileDataManager.OpenFiles(m_SubStreamNames, m_OpenMode,
                            m_IO.m_TransportsParameters,
                            m_BP4Serializer.m_Profiler.m_IsActive);
```
  -
      - Rank 0 creates the metadata files (`{name}.md.0` and `{name}.md.idx`)

  - BP4Writer::InitBPBuffer
      - Prepares the buffer headers
      - BP4 supports an Append mode where a simulation can be restarted from a given step (all the data that was written previously will be loaded before continuing the execution)
      - If the mode is not Append and the files are all new, the function makes the header for the data, matadata and the metadata index file by calling `MakeHeader`.

The `MakeHeader` function:
- Called by the data, metadata and the index metadata files
- Adds ADIOS and BP version to the header 
- Adds information about the format of the data (little endian .. )
- Adds the first `ProcessGroup` block in the data file following the format of the header described in the next section

```
    m_BP4Serializer.PutProcessGroupIndex(
        m_IO.m_Name, m_IO.m_HostLanguage,
        m_FileDataManager.GetTransportsTypes());

```

## Buffer Headers

BP4 File structure:
- `outpub.bp` folder containing:
  - `md.idx`: table with 64 byte long rows indexing the metadata file
  - `md.0`: file with metadata information for all variables (`global.md` in BP3)
  - `data.0`, `data.aggregation_step`, ..., `data.N`: data files, incorporating metadata interspersed with the data object


## Debugging ADIOS

Build ADIOS with the `-DCMAKE_BUILD_TYPE=DEBUG` flag.<br/>
Using VSCode to debug ADIOS, the following `launch.json file` uses llbd to go through the code:
```json
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(lldb) Launch",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/bin/bpCamWriteRead",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "lldb"
        }
    ]
}
```

If the process hangs during debugging or even if it successfully ends, there migh still be danggling processes in the background. 
In order to run another lldb process, these need to be killed: `ps aux | grep lldb` and `kill {pid}`.

