# ADIOS Streaming Modes

Consumers can receive either all the steps (all to all mode) or the producer can distribute steps amond the consumers (one step can only be consumed by one consumer) either in round robin order or on demand.

The `StepDistributionMode` input parameter for an ADIOS engine can be used for SST to distinguish between the streaming modes.

## Default

By default, the value is **AllToAll**, which
means that all timesteps are to be delivered to all readers (subject
to discard rules, etc.). Example using running two consumers for 6 steps.

```
$ ./sstWriter 100 1 & ./sstReader 100 1 p1 & ./sstReader 100 1 p2
p0: Put step 0
p0: Put step 1
p2: Get step 0
p1: Get step 0
p0: Put step 2
p1: Get step 1
p2: Get step 1
p1: Get step 2
p2: Get step 2
p0: Put step 3
p2: Get step 3
p0: Put step 4
p1: Get step 3
p0: Put step 5
p2: Get step 4
p1: Get step 4
SST,Write,1,100,1,6,440,4406
p2: Get step 5
p1: Get step 5
SST,Read,1,100,1,6,121,5276
SST,Read,1,100,1,6,114,4837
```

## Distributed steps

The `StepDistributionMode` controls how steps are distributed, particularly when there are
multiple readers. There are two cases:
- **RoundRobin**, each step is delivered
only to a single reader, determined in a round-robin fashion based
upon the number or readers who have opened the stream at the time the
step is submitted. In the example running two consumers for 6 steps, consumer 1 will receive steps 0, 2 and 4; consumer 2 will receive steps 1, 3 and 5.


```
$ ./sstWriter 100 1 & ./sstReader 100 1 p1 & ./sstReader 100 1 p2
p0: Put step 0
p0: Put step 1
p0: Put step 2
p1: Get step 0
p1: Get step 2
p0: Put step 3
p2: Get step 1
p0: Put step 4
p2: Get step 3
p1: Get step 4
p0: Put step 5
SST,Write,1,100,1,6,841,7681
SST,Read,1,100,1,3,168,9306
p2: Get step 5
SST,Read,1,100,1,3,154,8543
```

- **OnDemand** each step is delivered to a
single reader, but only upon request (with a request being initiated
by the reader doing BeginStep()).  Normal reader-side rules (like
BeginStep timeouts) and writer-side rules (like queue limit behavior) apply.

```
$ ./sstWriter 100 1 & ./sstReader 100 1 p1 & ./sstReader 100 1 p2
p0: Put step 0
p0: Put step 1
p0: Put step 2
p1: Get step 0
p0: Put step 3
p2: Get step 1
p1: Get step 2
p0: Put step 4
p2: Get step 2
p2: Get step 4
p0: Put step 5
SST,Write,1,100,1,6,436,3604
p2: Get step 5
SST,Read,1,100,1,2,62,4362
SST,Read,1,100,1,4,108,3990
```
