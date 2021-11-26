import numpy as np
import adios2
import sys

if len(sys.argv) < 2:
    print("Usage: %s array_size " %(sys.argv[0]))
    exit(1)

nx = int(sys.argv[1])
shape = [nx]
start = [0]
count = [nx]

# with-as will call adios2.close on fh at the end
# if only one rank is active pass MPI.COMM_SELF
with adios2.open("cfd.sst", "r", config_file="adios.xml", io_in_config_file="test") as fh:
    for fstep in fh:

    # inspect variables in current step
        step_vars = fstep.available_variables()

    # print variables information
        for name, info in step_vars.items():
            print("variable_name: " + name)
            for key, value in info.items():
                print("\t" + key + ": " + value)
        print("\n")

    # track current step
    step = fstep.current_step()
    if( step == 0 ):
      size_in = fstep.read("size")

    # read variables return a numpy array with corresponding selection
    pressure = fstep.read("pressure", start, count)
