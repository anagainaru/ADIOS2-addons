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
  - BP4Writer::InitTransports
  - BP4Writer::InitBPBuffer

## Buffer Headers

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

