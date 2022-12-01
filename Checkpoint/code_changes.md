# Code changes in ADIOS2 to use SCR

## API

In order for ADIOS to run with SCR, it needs to be build with SCR support.
```
cmake \
  -DADIOS2_USE_SCR=ON \
  -DADIOS2_USE_MPI=ON \
  -DSCR_ROOT=/path/to/scr-v3.0.1/install \
  ../ADIOS2
```

By default SCR is not used for data transfers even if ADIOS2 is build with SCR. The `UseSCR` parameter needs to be selected in order for SCR to do anything.
If the `UseSCR` flag is set, the SCR library needs to be initialized and finalized in the user code.

```c++
#include "scr.h"

MPI_Init();
SCR_Init();

adios2::IO io = adios.DeclareIO("TestIO");
io.SetParameters({{"UseSCR", "1"}});

adios2::Engine engine = io.Open(fname, adios2::Mode::Write);

for (step=0; step<total_steps; step++)
{
  engine.Put(var, data);
}

engine.Close();

SCR_Finalize();
MPI_Finalize();
```

SCR works only for MPI codes. The library needs to be initialized after `MPI_Init` and finalized before `MPI_Finalize`.
If the flag is set without SCR being initialized an error will occur.

```
SCR v3.0.0 ABORT: rank -2 on (null): Must call SCR_Init() before SCR_Start_output() @ /ccs/home/againaru/adios/ADIOS2-scr/scr-v3.0.1/scr/src/scr.c:3161
```

If SCR is initialized before MPI was initialized an error will occur.
```
*** The MPI_Comm_dup() function was called before MPI_INIT was invoked.
```

## Code changes in ADIOS2

Changes only in the BP4 Writer code. Create a bool variable `m_SCR` that stores the flag for the `UseSCR` flag that can be set by users to specify that the SCR library will be used for transfering the BP files.

```diff
diff --git a/source/adios2/engine/bp4/BP4Writer.h b/source/adios2/engine/bp4/BP4Writer.h
index 056d8d4fa..1fbc5dc44 100644
--- a/source/adios2/engine/bp4/BP4Writer.h
+++ b/source/adios2/engine/bp4/BP4Writer.h
@@ -65,6 +65,8 @@ private:
     /* transport manager for managing the metadata index file */
     transportman::TransportMan m_FileMetadataIndexManager;

+    bool m_SCR;
+
     /*
      *  Burst buffer variables
      */
diff --git a/source/adios2/engine/bp4/BP4Writer.cpp b/source/adios2/engine/bp4/BP4Writer.cpp
index bcb86792b..6a3570d52 100644
--- a/source/adios2/engine/bp4/BP4Writer.cpp
+++ b/source/adios2/engine/bp4/BP4Writer.cpp
@@ -199,6 +240,7 @@ void BP4Writer::InitParameters()
                                  "in call to BP4::Open to write");
     m_WriteToBB = !(m_BP4Serializer.m_Parameters.BurstBufferPath.empty());
     m_DrainBB = m_WriteToBB && m_BP4Serializer.m_Parameters.BurstBufferDrain;
+    m_SCR = helper::GetParameter(m_IO.m_Parameters, "UseSCR", m_Verbosity);
 }
```

Initialize and finalize SCR withing the Open and Close functions to the writer engine.

```diff
diff --git a/source/adios2/engine/bp4/BP4Writer.cpp b/source/adios2/engine/bp4/BP4Writer.cpp
index bcb86792b..6a3570d52 100644
--- a/source/adios2/engine/bp4/BP4Writer.cpp
+++ b/source/adios2/engine/bp4/BP4Writer.cpp
@@ -20,6 +20,10 @@
 #include <iostream>

+#ifdef ADIOS2_HAVE_SCR
+#include <scr.h>
+#endif
+
 namespace adios2
 @@ -144,6 +148,40 @@ void BP4Writer::Flush(const int transportIndex)
 }

 // PRIVATE
+#ifdef ADIOS2_HAVE_SCR
+    void InitSCR(const std::string fname){
+        SCR_Start_output(fname.c_str(), SCR_FLAG_CHECKPOINT);
+    }
+
+    void CloseSCR(const std::string fname){
+        int scr_valid = 1;
+        SCR_Complete_output(scr_valid);
+    }
+#endif

std::string SCRRouteFile(std::string name)
{
@@ -156,6 +194,9 @@ void BP4Writer::Init()
     }
     InitTransports();
     InitBPBuffer();
+#ifdef ADIOS2_HAVE_SCR
+    if (m_SCR) InitSCR(m_Name);
+#endif
 }
@@ -541,6 +599,9 @@ void BP4Writer::DoClose(const int transportIndex)
         m_FileDrainer.Finish();
     }
     // m_BP4Serializer.DeleteBuffers();
+#ifdef ADIOS2_HAVE_SCR
+    if (m_SCR) CloseSCR(m_Name);
+#endif
 }
```

Mark all the files inside the BP folder (data, metadata, index) to be handled by the SCR library

```diff
diff --git a/source/adios2/engine/bp4/BP4Writer.cpp b/source/adios2/engine/bp4/BP4Writer.cpp
index bcb86792b..6a3570d52 100644
--- a/source/adios2/engine/bp4/BP4Writer.cpp
+++ b/source/adios2/engine/bp4/BP4Writer.cpp
 @@ -144,6 +148,40 @@ void BP4Writer::Flush(const int transportIndex)
 }

+std::string SCRRouteFile(std::string name)
+{
+#ifdef ADIOS2_HAVE_SCR
+    char scr_name[SCR_MAX_FILENAME];
+    SCR_Route_file(name.c_str(), scr_name);
+
+    std::string s(scr_name);
+    return s;
+#else
+    return name;
+#endif
+}
+
+std::vector<std::string> AddSCRRouteInfo(const std::vector<std::string> files)
+{
+    std::vector<std::string> newFiles;
+    for (const auto &name : files)
+    {
+        newFiles.push_back(SCRRouteFile(name));
+    }
+    return newFiles;
+}
@@ -199,6 +240,7 @@ void BP4Writer::InitParameters()
                                  "in call to BP4::Open to write");
     m_WriteToBB = !(m_BP4Serializer.m_Parameters.BurstBufferPath.empty());
     m_DrainBB = m_WriteToBB && m_BP4Serializer.m_Parameters.BurstBufferDrain;
+    m_SCR = helper::GetParameter(m_IO.m_Parameters, "UseSCR", m_Verbosity);
 }

 void BP4Writer::InitTransports()
@@ -241,21 +283,28 @@ void BP4Writer::InitTransports()
                 m_BP4Serializer.m_RankMPI);
             m_FileDrainer.Start();
         }
+        if (m_SCR)
+        {
+            m_SubStreamNames = AddSCRRouteInfo(m_SubStreamNames);
+        }
     }
```
