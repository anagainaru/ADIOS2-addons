# Streaming on demand

Using a configuration file 

Compiling

```
cmake -D adios2_ROOT=/Users/95j/work/adios/ADIOS2-main/install -D MPI_ROOT=/Users/95j/opt/usr/local -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc ..
make -j4
```

Results when using one or two consumers
```
$ ./sstWriter 100 2 & ./sstReader 100 2
SST,Write,1,100,2,100,6976,56482
SST,Read,1,100,2,98,2026,57045

$ ./sstWriter 100 2 & ./sstReader 100 2 & ./sstReader 100 2
SST,Write,1,100,2,100,6551,56033
SST,Read,1,100,2,2,177,57931
SST,Read,1,100,2,97,2008,56521
```

For some reason not all steps are read. In the previous example the number of steps writen/read by the producer/consumer are the 6th element in the CSV data (100/2/97). Each process is plotting the steps it puts/gets for one variable per step and 6 steps:

```
./sstWriter 100 1 & ./sstReader 100 1 p1 & ./sstReader 100 1 p2
p0: Put step 0
p0: Put step 1
p0: Put step 2
p1: Get step 0
p0: Put step 3
p1: Get step 2
p2: Get step 1
p0: Put step 4
p2: Get step 2
p0: Put step 5
SST,Write,1,100,1,6,731,7499
SST,Read,1,100,1,2,135,9184
p2: Get step 5
SST,Read,1,100,1,3,286,8430
```
Step 3 is not read by any of the consumers in the previous example.


## Implementation

**Issue 1** 

For AlltoAll or RoundRobin:

- When a timestep is created on the writer, the metadata for it gets sent to every reader (and queued there if it’s not immediately needed).  Then when the writer does Close(), it also sends a close message to every reader noting it’s new status.  The reader then knows that it’s at end-of-stream when 
   1) it has no more metadata, and 
   2) it has received a close message so it knows the writer won’t be producing more timesteps. That’s safe because the metadata messages are guaranteed to arrive before the close message.
 
With OnDemand:
- By default we send no metadata to anyone, but instead as soon as the reader enters BeginStep we send a TimestepRequest message to the writer.  
- Then we fall into the more standard BeginStep handling (specifically `SstAdvanceStepMin()` in `sst/cp/cp_reader.c`).  That is, we look to see if we have any metadata queued and if not, we wait for it.  OR we return EndOfStream if we have no more metadata and we have received a close message from the writer.  
- in the OnDemand case this is a race condition because the writer might send us more metadata (for example for the last step) in response to that OnDemand message.  If that metadata arrives before we look for it in `SstAdvanceStepMin()`, we’ll get all the steps and everything will be fine.  If it doesn’t, we’ll see that there’s no metadata queued and the writer has already send a close message, so we conclude EndOfStream.
 
**Issue 2** 

We send metadata to the reader in two places in the code, 
- when a new request message has just come in and we have metadata queued,
- when new metadata has just come in from the writer and we have requests queued.  

In one of those cases the line of code that updates the last timestep that had been sent on demand was missing.  That produced a different race condition.  * A timestep sent there might be sent just once because once it’s been consumed by the reader, we’ll get a release message and it will be removed from the queue, but since it’s not removed until we get that release message, we might send it again if we get another TimestepRequest message while it’s still in the queue.*

**Multiple reader error**
```
$ ./sstWriter 100 10 writer & ./sstReader 100 reader1 &  ./sstReader 100 reader2 & ./sstReader 100 reader3
SST,Write,1,100,1,10,622,6589
SST,Read,1,100,1,4,97,7558
SST,Read,1,100,1,4,106,104326
SST,Read,1,100,1,3,85,7211
```
