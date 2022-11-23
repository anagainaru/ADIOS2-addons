# ADIOS-2 with Checkpointing capabilities

## Scalable Checkpoint Restart (SCR) library

Pull request: [https://github.com/ornladios/ADIOS2/pull/3294](https://github.com/ornladios/ADIOS2/pull/3294)

Most SCR calls are placed in the application.  For an application, see `examples/hello/bpWriter/helloBPWriter.c` first.  I started with it and added some comments to describe the SCR interface.  When testing restart, I realized that `helloBPWriter.c` has no corresponding Reader, so I then added SCR calls to `helloBPWriter.cpp` and `helloBPReader.cpp`.  The SCR calls in the application start up and shut down the SCR library, and they add the start/end bookends that define boundaries of the checkpoint and restart phases.

`SCR_Route_file` is called from the BP4 engine.  This should be called for each physical file that ADIOS writes.  SCR manages directories, so I've commented out the `mkdir` operations for now.  During a write phase, `SCR_Route_file` tells SCR that the file belongs to the active dataset and it provides the path where the file should (eventually) be written to on the parallel file system.  As output from this call, SCR provides a path to where the file should be written instead.  This output path may be a temporary location like `/dev/shm` or a node-local SSD depending on how SCR was configured.

### Build

The SCR library used: [https://github.com/LLNL/scr/releases/download/v3.0.1/scr-v3.0.1.tgz](https://github.com/LLNL/scr/releases/download/v3.0.1/scr-v3.0.1.tgz)

```
module load cmake/3.23.1

cmake -DCMAKE_INSTALL_PREFIX=/path/to/scr-v3.0.1/install -DCMAKE_BUILD_TYPE=Debug -DSCR_RESOURCE_MANAGER=LSF  -DENABLE_PDSH=OFF  ..
make -j4
make -j4 install

module load gcc
cmake \
  -DCMAKE_INSTALL_PREFIX=/path/to/ADIOS2-scr/install \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_VERBOSE_MAKEFILE=ON \
  -DADIOS2_USE_MPI=ON \
  -DADIOS2_USE_HDF5=OFF \
  -DADIOS2_USE_CUDA=OFF \
  -DADIOS2_USE_Fortran=OFF \
  -DADIOS2_USE_Python=OFF \
  -DSCR_ROOT=/path/to/scr-v3.0.1/install \
  ../ADIOS2
make -j4
make -j4 install
```

### Run

```
#!/bin/bash -l
#BSUB -P $project_number
#BSUB -W 00:01
#BSUB -nnodes 1
#BSUB -J adiosSCR
#BSUB -o scr.out.%J
#BSUB -e scr.out.%J

module load gcc

export SCR_USER_NAME={summit_username}

jsrun -r2 ./bin/hello_bpWriter_mpi
jsrun -r2 ./bin/hello_bpReader_mpi
```

### Other parameters

By default, SCR uses "cache bypass" so that each file is written directly to the file system.  That is, `SCR_Route_file` just returns the original path that the user wanted to write to.  It does not return a temporary path.  This "cache bypass" mode should work on any system, including those that do not have sufficient temporary storage.  To configure SCR to write to node local storage, disable "cache bypass" mode:
```
export SCR_CACHE_BYPASS=0
```

When running with cache enabled, one should see ADIOS files in subdirectories within `/dev/shm/` like `/dev/shm/$USER/scr.<jobid>/scr.dataset.<id>`.  The files will have also been copied to the parallel file system when the Writer application exits.  By default, SCR flushes any cached checkpoint during `SCR_Finalize()`.

It can be useful to run tests where SCR does not flush.  One can use this mode to simulate a restart after a failure, in which case the SCR library may not have had a chance to flush the checkpoint:
```
export SCR_FLUSH=0
```

With this, SCR will not flush files during `SCR_Finalize()`.

To demonstrate, delete the directory from the parallel file system and run the Writer again.  This time, the files will be in `/dev/shm` but not on the parallel file system.  If one then runs the Reader, the SCR library flushes the files from `/dev/shm` to the parallel file system during the call to `SCR_Init()`.

This is because I have specified `SCR_GLOBAL_RESTART=1` in the Reader code.  This tells SCR to rebuild and flush any cached dataset during `SCR_Init()`.  Not all applications need to do this, but for now this is required in order to maintain the directory structure that ADIOS2 expects to see when inspecting the files for reading.

SCR uses `/dev/shm` as its default cache, since it is fast and available on any Linux cluster.  One can specify another location by setting the "cache base", e.g., to point to a node-local SSD:
```
export SCR_CACHE_BASE=/mnt/ssd
```

## Correctness test

## Performance test
