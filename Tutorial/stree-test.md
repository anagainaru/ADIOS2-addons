# Benchmark to stree the filesystem

```
Can you add these 2 small adios2_iotest apps to your workflow, running at the same time of course.
 
app 1 writes 96MB/process every 5 seconds
app2 reads them and writes them, and sleeps 1 second between successful steps
 
The decomposition of processes should be the following:
Run each on N processes where N is even number
 
mpirun -n N  adios2_iotest -a 1 -x pipe.xml -c pipe.cfg -t -w  -d N  1  1
mpirun -n N  adios2_iotest -a 2 -x pipe.xml -c pipe.cfg -t -w  -d N/2  2  1
```
