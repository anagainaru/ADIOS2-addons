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

### Run in debug mode

Set the `SstVerbose` environmental variable to 5 (levels 1 to 5 each increase the verbosity of ADIOS).

```
$ export SstVerbose=5
$ ./sstWriter 100 2 2>> sst.log & ./sstReader 100 2 p1 2>> sst.log & ./sstReader 100 2 p2 2>> sst.log
[8] 53196
[9] 53197
p2: Get step 0 variable0 0
p2: Get step 0 variable1 1
p1: Get step 1 variable0 2
p1: Get step 1 variable1 3
p2: Get step 2 variable0 4
p2: Get step 2 variable1 5
p1: Get step 2 variable0 4
p1: Get step 2 variable1 5
p1: Get step 4 variable0 8
p1: Get step 4 variable1 9
SST,Write,1,100,2,6,987,10595
SST,Read,1,100,2,2,168,13882
p1: Get step 5 variable0 10
p1: Get step 5 variable1 11
SST,Read,1,100,2,4,234,12251
```

Step 2 read by both readers
```
Writer 0 (0x7f8ede42aaa0): In RequestStepHandler, trying to send TS 2, examining TS 2
Writer 0 (0x7f8ede42aaa0): Sending Queued TimestepMetadata for timestep 2, reference count = 0
Writer 0 (0x7f8ede42aaa0): Sent timestep 2 to reader cohort 0
Writer 0 (0x7f8ede42aaa0): ADDING timestep 2 to sent list for reader cohort 0, READER 0x7f8eee437800, reference count is now 1
Writer 0 (0x7f8ede42aaa0): PRELOADMODE for timestep 2 non-default for reader , active at timestep 0, mode 1
DP Writer 0 (0x7f8ede42aaa0): Per reader registration for timestep 2, preload mode 1
DP Writer 0 (0x7f8ede42aaa0): Sending Speculative Preload messages, reader 0x7f8eee436160, timestep 2
Writer 0 (0x7f8ede42aaa0): Sending a message to reader 0 (0x7f9cf9740510)
DP Reader 0 (0x7f9cf9740510): Got a preload message from writer rank 0 for timestep 2, fprint 90beed80785d272
Reader 0 (0x7f9cf9740510): Received a Timestep metadata message for timestep 2, signaling condition
DP Reader 0 (0x7fa407526810): Satisfying remote memory read with preload from writer rank 0 for timestep 1, fprint 90beed80785d272
Reader 0 (0x7f9cf9740510): DP Reader 0 (0x7fa407526810): Examining metadata for Timestep 2
Satisfying remote memory read with preload from writer rank 0 for timestep 1, fprint 90beed80785d272
Reader 0 (0x7f9cf9740510): Returning metadata for Timestep 2
Reader 0 (0x7f9cf9740510): Setting TSmsg to Rootentry value
DP Reader 0 (0x7f9cf9740510): EVPATH registering reader arrival of TS 2 metadata, preload mode 1
Reader 0 (0x7f9cf9740510): SstAdvanceStep returning Success on timestep 2

Writer 0 (0x7f8ede42aaa0): In RequestStepHandler, trying to send TS 2, examining TS 3
Writer 0 (0x7f8ede42aaa0): In RequestStepHandler, trying to send TS 2, examining TS 2
Writer 0 (0x7f8ede42aaa0): Sending Queued TimestepMetadata for timestep 2, reference count = 1
Writer 0 (0x7f8ede42aaa0): Sent timestep 2 to reader cohort 1
Writer 0 (0x7f8ede42aaa0): ADDING timestep 2 to sent list for reader cohort 1, READER 0x7f8ede436d30, reference count is now 2
Writer 0 (0x7f8ede42aaa0): PRELOADMODE for timestep 2 non-default for reader , active at timestep 0, mode 1
DP Writer 0 (0x7f8ede42aaa0): Per reader registration for timestep 2, preload mode 1
DP Writer 0 (0x7f8ede42aaa0): Sending Speculative Preload messages, reader 0x7f8ede436e60, timestep 2


DP Reader 0 (0x7fa407526810): Writer 0 (0x7f8ede42aaa0): Got a preload message from writer rank 0 for timestep 2, fprint 90beed80785d272
QueueMaintenance complete
Writer 0 (0x7f8ede42aaa0): Received a release timestep message for timestep 2 from reader cohort 0
Writer 0 (0x7f8ede42aaa0): Got the lock in release timestep
Writer 0 (0x7f8ede42aaa0): Reader 0 (0x7fa407526810): Doing dereference sent
Received a Timestep metadata message for timestep 2, signaling condition
Writer 0 (0x7f8ede42aaa0): Reader sent timestep list 0x7f8eee43cd80, trying to release 2
Reader 0 (0x7fa407526810): Examining metadata for Timestep 2
Writer 0 (0x7f8ede42aaa0): Reader 0 (0x7fa407526810): Reader considering sent timestep 2,trying to release 2
Returning metadata for Timestep 2
Writer 0 (0x7f8ede42aaa0): Reader 0 (0x7fa407526810): SubRef : Writer-side Timestep 2 now has reference count 1, expired 0, precious 0

Reader 0 (0x7fa407526810): Writer 0 (0x7f8ede42aaa0): SstAdvanceStep returning Success on timestep 2

Writer 0 (0x7f8ede42aaa0): Reader sent timestep list 0x7f8eee43b020, trying to release 2
Writer 0 (0x7f8ede42aaa0): Reader considering sent timestep 2,trying to release 2
Writer 0 (0x7f8ede42aaa0): SubRef : Writer-side Timestep 2 now has reference count 1, expired 0, precious 0

Writer 0 (0x7f8ede42aaa0): Reader 0 status Established has last released 2, last sent 2
Writer 0 (0x7f8ede42aaa0): Reader 1 status Established has last released 2, last sent 2
Writer 0 (0x7f8ede42aaa0): QueueMaintenance, smallest last released = 2, count = 2
Writer 0 (0x7f8ede42aaa0): Writer tagging timestep 2 as expired

Writer 0 (0x7f8ede42aaa0): IN TS WAIT, ENTRIES are Timestep 2 (exp 1, Prec 0, Ref 1), Count now 4
```

Step 3 not read by any readers
```
DP Writer 0 (0x7f8ede42aaa0): ProvideTimestep, registering timestep 3, data 0x7f8ebe818600, fprint 90beed80785d272
Writer 0 (0x7f8ede42aaa0): Sending TimestepMetadata for timestep 3 (ref count 1), one to each reader
Writer 0 (0x7f8ede42aaa0): SubRef : Writer-side Timestep 3 now has reference count 0, expired 0, precious 0

Writer 0 (0x7f8ede42aaa0): IN TS WAIT, ENTRIES are Timestep 3 (exp 0, Prec 0, Ref 0), Count now 4

Writer 0 (0x7f8ede42aaa0): Writer tagging timestep 3 as expired
DP Writer 0 (0x7f8ede42aaa0): Releasing timestep 3
Remove queue Entries removing Timestep 3 (exp 1, Prec 0, Ref 0), Count now 2
```

Differences between steps
```
$ cat sst.log | grep "Sending Speculative Preload messages"
DP Writer 0 (0x7f8ede42aaa0): Sending Speculative Preload messages, reader 0x7f8eee436160, timestep 0
DP Writer 0 (0x7f8ede42aaa0): Sending Speculative Preload messages, reader 0x7f8ede436e60, timestep 1
DP Writer 0 (0x7f8ede42aaa0): Sending Speculative Preload messages, reader 0x7f8eee436160, timestep 2
DP Writer 0 (0x7f8ede42aaa0): Sending Speculative Preload messages, reader 0x7f8ede436e60, timestep 2
DP Writer 0 (0x7f8ede42aaa0): Sending Speculative Preload messages, reader 0x7f8eee436160, timestep 2
DP Writer 0 (0x7f8ede42aaa0): Sending Speculative Preload messages, reader 0x7f8ede436e60, timestep 4
DP Writer 0 (0x7f8ede42aaa0): Sending Speculative Preload messages, reader 0x7f8ede436e60, timestep 5

$ cat sst.log | grep "Received a Timestep metadata message for timestep"
Reader 0 (0x7f9cf9740510): Received a Timestep metadata message for timestep 0, signaling condition
Reader 0 (0x7fa407526810): Received a Timestep metadata message for timestep 1, signaling condition
Reader 0 (0x7f9cf9740510): Received a Timestep metadata message for timestep 2, signaling condition
Received a Timestep metadata message for timestep 2, signaling condition
Reader 0 (0x7f9cf9740510): Received a Timestep metadata message for timestep 2, signaling condition
Reader 0 (0x7fa407526810): Received a Timestep metadata message for timestep 4, signaling condition
Reader 0 (0x7fa407526810): Writer 0 (0x7f8ede42aaa0): Received a Timestep metadata message for timestep 5, signaling condition
```
