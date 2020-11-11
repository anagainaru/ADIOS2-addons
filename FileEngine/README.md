# File Engines

Campaign management having hierarchies of files each with its own metadata. There needs to be an overall metadata to hold all the files connected to a campaign.

### Description of the current file engines

The file engine used by ADIOS applications is BP4.
* Examplain the code for BP4 (The BP4 engine works in steps etc)
* What would a new engine need to implement

### Adding a new storage engine in ADIOS

**Add the new read/write functionality**

Create a new folder in `${ADIOS_HOME}/source/adios2/engine/` to include the code used by the new engine (in our cases this will be called `campb`).
Add the new files in the path used by the compiler.

In `${ADIOS_HOME}/source/adios2/CMakeLists.txt` two lines need to be added for the reader and the writer sides of the engine:

```
engine/cambp/CamBPReader.cpp engine/cambp/CamBPReader.tcc
engine/cambp/CamBPWriter.cpp engine/cambp/CamBPWriter.tcc
```

**Create the mapping between the string name used for the engine and the pointer to the classes that implement the engnine.**

In `${ADIOS_HOME}/source/adios2/core/IO.cpp`, add a new entry in the Factory data structure.

```
#include "adios2/engine/cambp/CamBPReader.h"
#include "adios2/engine/cambp/CamBPWriter.h"

std::unordered_map<std::string, IO::EngineFactoryEntry> Factory = {
   {"cambp",
    {IO::MakeEngine<engine::CamBPReader>, IO::MakeEngine<engine::CamBPWriter>}},
}
```

**Deactivate the streaming capability for the new engine**

In `${ADIOS_HOME}/source/utils/adios_reorganize/Reorganize.cpp` on line 275 add the new file engine:

```
    if (s == "file" || s == "bpfile" || s == "cambp" ||
        s == "bp3" || s == "hdf5")
    {
         handleAsStream = false;
    }
```

**Add the new engine in the printUsage function**

In `${ADIOS_HOME}/source/utils/adios_reorganize/Reorganize.cpp` on line 295:

```
Supported read methods: BPFile, CamBP, HDF5, SST, DataMan,
```

**Add the new engine to the list returned by the getEngineList**

In the `${ADIOS_HOME}/source/utils/bpls/bpls.cpp` in the getEnginesList function, add the new engine to the returned list.

```
     {
         list.push_back("HDF5");
         list.push_back("BPFile");
+        list.push_back("CamBP");
     }
     else
     {
         list.push_back("BPFile");
+        list.push_back("CamBP");
         list.push_back("HDF5");
     }
 #else
     list.push_back("BPFile");
+    list.push_back("CamBP");
 #endif
     return list;
 }
```

### Example using the new engine

Simple example to test the basic functionality is implemented in the `campaign_example` folder.
In order for this example to be included in ADIOS, a new folder needs to be created for it in ${ADIOS_ROOT}/examples (in our case `campaign`) and the CMakeLists.txt file needs to be updated with the following line:

```
add_subdirectory(campaign)
```

Once ADOIS is build in a folder `build` the new example code can be executed from the bin directory.

```
$ cd ${ADIOS_HOME}/build
$ make -j
$ ./bin/bpCamWriteRead
Steps expected by the reader: 10
Rank 0 expects 100 elements
Simualation step 0 : 100 elements: 10
Simualation step 10 : 100 elements: 20
Simualation step 20 : 100 elements: 30
Simualation step 30 : 100 elements: 40
Simualation step 40 : 100 elements: 50
Simualation step 50 : 100 elements: 60
Simualation step 60 : 100 elements: 70
Simualation step 70 : 100 elements: 80
Simualation step 80 : 100 elements: 90
Simualation step 90 : 100 elements: 100
```

The example is based on the WriteRead testing class in the `testing` folder in the ADIOS root directory 
([Link to Github](https://github.com/ornladios/ADIOS2/blob/master/testing/adios2/engine/bp/TestBPWriteReadADIOS2.cpp)).

```
for i in bin/Test.Engine.BP.WriteRead*; do echo $i; ./$i; done > ${ADIOS_HOME}/testWriteRead.output
```

Example output for the WriteRead test code:
```
$  ./bin/Test.Engine.BP.WriteReadADIOS2.Serial

[==========] Running 8 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 8 tests from BPWriteReadTestADIOS2
[ RUN      ] BPWriteReadTestADIOS2.ADIOS2BPWriteRead1D8
[       OK ] BPWriteReadTestADIOS2.ADIOS2BPWriteRead1D8 (5 ms)
[ RUN      ] BPWriteReadTestADIOS2.ADIOS2BPWriteRead2D2x4
[       OK ] BPWriteReadTestADIOS2.ADIOS2BPWriteRead2D2x4 (2 ms)
[ RUN      ] BPWriteReadTestADIOS2.ADIOS2BPWriteRead2D4x2
[       OK ] BPWriteReadTestADIOS2.ADIOS2BPWriteRead2D4x2 (2 ms)
[ RUN      ] BPWriteReadTestADIOS2.ADIOS2BPWriteRead10D2x2
[       OK ] BPWriteReadTestADIOS2.ADIOS2BPWriteRead10D2x2 (3 ms)
[ RUN      ] BPWriteReadTestADIOS2.ADIOS2BPWriteRead2D4x2_ReadMultiSteps
[       OK ] BPWriteReadTestADIOS2.ADIOS2BPWriteRead2D4x2_ReadMultiSteps (3 ms)
[ RUN      ] BPWriteReadTestADIOS2.ADIOS2BPWriteRead2D4x2_MultiStepsOverflow
[       OK ] BPWriteReadTestADIOS2.ADIOS2BPWriteRead2D4x2_MultiStepsOverflow (1 ms)
[ RUN      ] BPWriteReadTestADIOS2.OpenEngineTwice
[       OK ] BPWriteReadTestADIOS2.OpenEngineTwice (2 ms)
[ RUN      ] BPWriteReadTestADIOS2.ReadStartCount
[       OK ] BPWriteReadTestADIOS2.ReadStartCount (1 ms)
[----------] 8 tests from BPWriteReadTestADIOS2 (19 ms total)

[----------] Global test environment tear-down
[==========] 8 tests from 1 test suite ran. (20 ms total)
[  PASSED  ] 8 tests.
```
