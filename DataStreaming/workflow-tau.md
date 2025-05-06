## Monitoring perfromance

Install TAU with ADIOS support. For example on Perlmutter:
```
git clone git@github.com:UO-OACISS/tau2.git
cd tau2/
./configure -prefix=/path/to/install/tau -cc=cc -c++=CC -fortran=ftn -bfd=download -openmp -mpi -python3 -adios=/path/to/install/adios2-gcc11.2.0/
 ```

Testing tau_exec with one of the ADIOS test programs:
```
srun -n 1 tau_exec -adios2 -T mpi,adios2 /path/to/install/adios2-gcc11.2.0/bin/adios2_hello_bpWriter_mpi
 ```

There should be a tauprofile-adios2_hello_bpWriter_mpi.bp file created. 

Note, the TAU plugin only works with ADIOS MPI programs (ADIOS_Init from TAU occurs with the MPI_Init).
 
Perfstubs can be used to add additional instrumentation, such as to add timers or counters. 

Example for WDMApp, for key subroutines here: https://github.com/UO-OACISS/perfstubs/blob/master/perfstubs_api/README.md
 
I’ll also note that if you’re running Python, the perfstubs module can be installed to automatically collect things like the function call times, without explicit instrumentation.

```
git clone --recurse-submodules https://github.com/UO-OACISS/perfstubs.git
cd perfstubs
pip install --user --editable .
```

For instance, for a simple program `test.py` that just runs one function (Test) to print the MPI Rank, TAU will automatically time Test().
```
tau_exec -adios2 -T mpi,openmp,gnu,adios2 python3 -m pstubs test.py
 ```
